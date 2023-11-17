import numpy as np
from instance_data_ss import data
import matplotlib.pyplot as plt
import control

"""
Formulate state space model for the supply chain of the form: 
        x(k+1) = Ax(k) + Bu(k)  
        y(k) = Cx(k) + Du(k) 
This code generates matrices A, B, C, D for given instance data 

Notes: 
    - The state space might need to be modified to x(k+1) = Ax(k) + Bu(k) + B'u(k-t) for a inputs with time delay t 
    can try to generalize results from systems without time delay to time-delayed instances. 
    - Might need revisions on how to define the state space and compute the associated matrices when we have more than 
    one allowed temperature setting 
"""

# User options for printing and plotting results
verbose = True
plotting = True

# Vector of symbolic state variables
x_symb = []
# Initial state
x0 = []
# State costs
c_x = []

# Generating inventory variables (I_i,q,k,t)
for i in data['P_set']+data['D_set']+data['R_set']:
    for k in range(1, data['temperature']+1):
        for q in range(1, data['quality'] + 1):
            x_symb.append('I_'+i+'_'+str(q)+'_'+str(k))

            c_x.append(data['holding_costs'][i])

            if q == np.ceil(data['quality']):
                x0.append(50)
            else:
                x0.append(0)

# # Generating backorder variables (BO_i,t)
for i in data['R_set']:
    x_symb.append('BO_' + i)
    x0.append(0)
    c_x.append(data['backorder_penalty'])

for i in data['R_set']:
    x0.append(40)
    c_x.append(0)

# Vector of symbolic inputs
u_symb = []

# Input costs
c_u = []

# Generating production variables p_i,q,t
for i in data['P_set']:
    for q in range(1, data['quality'] + 1):
        u_symb.append('p_'+i+'_'+str(q))
        c_u.append(data['production_costs'][i])


# Generating sales variables s_i,j,q,k,t
for (i,j) in data['A_routes']:
    for k in range(1, data['temperature'] + 1):
        for q in range(1, data['quality'] + 1):
            u_symb.append('s_'+i+'_'+j+'_'+str(q)+'_'+str(k))
            c_u.append(data['transportation_costs'][(i,j)])

# Generating sales variables r_i,q,t
for i in data['R_set']:
    for q in range(1, data['quality'] + 1):
        u_symb.append('r_' + i + '_' + str(q))
        c_u.append(0)

# Vector of disturbances
d_symb = ['d_'+i for i in data['R_set']]

n_x = len(x_symb)
n_u = len(u_symb)
n_d = len(d_symb)

if verbose:
    print('\n')
    print('States: ', n_x)
    print('Inputs: ', n_u)
    print('Disturbances: ', n_d)
    print('Matrix dimensions:')
    print('    A:', n_x,'x', n_x)
    print('    B:', n_x,'x',n_u)
    print('    C:', 'n_y','x',n_x)
    print('    D:', 'n_y','x',n_u)
    print('\n')

# MATRIX A GENERATION
A = np.zeros((len(x_symb),len(x_symb)))
A_row = 0
A_col = 0
# Inventory
for i in data['P_set']+data['D_set']+data['R_set']:
    for k in range(1, data['temperature'] + 1):
        for q in range(1, data['quality'] + 1):
            if q + data['degradation'][k] <= data['quality'] and q >= data['minimum_quality'][i]:
                A[A_row, A_col + data['degradation'][k]] = 1
            A_row += 1
            A_col += 1
# # # Backorder
A_row = n_x-1
A_col = n_x-1
for i in data['R_set']:
    A[A_row, A_col] = 1
    A_row -= 1
    A_col -= 1


# MATRIX B GENERATION
B = np.zeros((len(x_symb),len(u_symb)))
# Iterating over all states
B_row = 0
for i in data['P_set']+data['D_set']+data['R_set']:
    for k in range(1, data['temperature'] + 1):
        for q in range(1, data['quality'] + 1):
        # Iterating over all inputs
            # Production
            B_col = 0
            for i_u in data['P_set']:
                for q_u in range(1, data['quality'] + 1):
                    if i == i_u and q == q_u:
                        B[B_row, B_col] = 1
                    B_col +=1
            # Shipment
            for (i_u, j_u) in data['A_routes']:
                for k_u in range(1, data['temperature'] + 1):
                    for q_u in range(1, data['quality'] + 1):

                        if i == i_u and q == q_u:
                            B[B_row, B_col] = -1
                        if i == j_u and q == q_u and q_u < data['quality']:
                            if j_u== 'R1' and q == 2:
                                B[B_row, B_col + data['degradation'][k_u] ] = 0
                            else:
                                B[B_row, B_col + data['degradation'][k_u] ] = 1
                        B_col += 1
            # Sales
            B_col_sales  = B_col
            for i_u in data['R_set']:
                for q_u in range(1, data['quality'] + 1):

                    if i == i_u and q == q_u:
                        B[B_row, B_col] = -1
                    B_col += 1

            B_row += 1


B_col = B_col_sales
for i in data['R_set']:
    B_col = B_col_sales
    for i_u in data['R_set']:
        for q_u in range(1, data['quality'] + 1):

            if i == i_u:
                B[B_row, B_col] = -1
            B_col += 1

    B_row += 1



# MATRICES B_d

# for i in range(n_x):
#     print(x_symb[i], [i for i in B[i,:] if i != 0])

B_d = np.zeros((n_x,n_d))
B_d_row = n_x-1
B_d_col = n_d-1
for _ in d_symb:
    B_d[B_d_row,B_d_col] = 1
    B_d_row -= 1
    B_d_col -= 1

# for i in range(n_x):
#     print(x_symb[i], [i for i in B[i,:] if i != 0])

# Observability analysis
eig_A,_ = np.linalg.eig(A)
I = np.identity(n_x)

# Matrix C
# Perfect observability
C = np.identity(n_x)

# No observability
C = np.zeros((n_x,n_x))

# Custom
C = np.zeros((6,n_x))
C[0][0] = 1
C[1][1] = 1
C[2][2] = 1
C[3][3:6] = [1,1,1]
C[4][6:9] = [1,1,1]
C[3][9] = 1

C = np.zeros((4,n_x))
C[0][0:3] = [1,1,1]
C[1][3:6] = [1,1,1]
C[2][6:9] = [1,1,1]
C[3][9] = 1


print(2)
##############################   OBSERVABILITY MATRIX    ##############################
C_mat = control.ctrb(A,B)
O_mat = control.obsv(A, C)
if verbose:
    print('Rank controllability matrix for (A,B): ', np.linalg.matrix_rank(C_mat))
    print('Rank observability matrix for (A,C): ', np.linalg.matrix_rank(O_mat))



##############################   FINAL FORM OF  A & B    ##############################

# A = np.concatenate((np.concatenate((A, B_d), axis = 1),
#                     np.concatenate((np.zeros((n_d, n_x)), np.eye(n_d)), axis = 1)), axis = 0)
#
# B = np.concatenate((B, np.zeros((n_d, n_u))), axis = 0 )
#
# C = np.concatenate((C, C_d), axis = 1)

##############################   PLOTTING A & B    ##############################




if plotting:

    fig, axs = plt.subplots(1,2, sharey = True, gridspec_kw={'width_ratios': [1, len(u_symb)/len(x_symb)]})

    ax = axs[0]
    ax.plot([0, 1], [1, 0], '--k', transform=ax.transAxes)
    ax.spy(A, markersize=4, color='k')
    ax.axhline(y=-0.5, color='lightgrey', linestyle='-', linewidth=3)
    ax.axvline(x=-0.5, color='lightgrey', linestyle = '-', linewidth=3)
    for i in range(1,11):
        if i%3==0:
            ax.axvline(x=-0.5 + i, color='lightgrey', linestyle='-', linewidth=1)
            ax.axhline(y=-0.5 + i, color='lightgrey', linestyle='-', linewidth=1)
        else:
            ax.axvline(x=-0.5+i, color='lightgrey', linestyle='--', linewidth = 0.75)
            ax.axhline(y=-0.5 + i, color='lightgrey', linestyle='--', linewidth=0.75)

    ax.axvline(x=-0.5 + i, color='lightgrey', linestyle='-', linewidth=3)
    ax.axhline(y=-0.5 + i, color='lightgrey', linestyle='-', linewidth=3)

    # ax.axvspan(xmin=-0.5, xmax=2.5, ymin = 2.5, ymax=-0.5,color='C0', alpha = 0.4)

    # ax.set_title('Rank A: '+ str(np.linalg.matrix_rank(A)))
    # ax.set_title('A')
    # ax.xaxis.tick_bottom()
    ax.axis('off')
    # ax.set_xticks(np.arange(1,9,2))
    # ax.set_xlim([-0.1,9.1])


    # plt.spy(O_1, markersize=1, color='k'); plt.title(r'$\Delta q_k = 1$ -- Rank matrix $\mathcal{O}_1$ =' + str(np.linalg.matrix_rank(O_1))); plt.tight_layout();

    ax = axs[1]

    ax.spy(B, markersize=4, color='k')
    ax.axhline(y=-0.5, color='lightgrey', linestyle='-', linewidth=3)
    ax.axvline(x=-0.5, color='lightgrey', linestyle='-', linewidth=3)
    for i in range(1, 11):
        linestyle_str = '--'
        ax.axhline(y=-0.5 + i, color='lightgrey', linestyle=linestyle_str, linewidth=1)
        ax.axvline(x=-0.5 + i, color='lightgrey', linestyle=linestyle_str, linewidth=1)

        if i%3 ==0:
            ax.axhline(y=-0.5 + i, color='lightgrey', linestyle='-', linewidth=1)

    ax.axhline(y=-0.5 + i, color='lightgrey', linestyle='-', linewidth=3)

    for i in range(11, 13):
        ax.axvline(x=-0.5 + i, color='lightgrey', linestyle=linestyle_str, linewidth=1)

    ax.axvline(x=-0.5 + i, color='lightgrey', linestyle='-', linewidth=3)
    ax.axvline(x=2.5, color='lightgrey', linestyle='-', linewidth=1)
    ax.axvline(x=8.5, color='lightgrey', linestyle='-', linewidth=1)

    print(fig.get_size_inches())
    ax.axis('off')
    # ax.set_title('B')
    # ax.set_title('Rank B: '+ str(np.linalg.matrix_rank(B)))


    plt.tight_layout()
    size = fig.get_size_inches()
    plt.draw()
    plt.show()

    # plt.figure()
    # plt.spy(C, markersize=0.5, color = 'k')
    # plt.show()

    # plt.figure()
    # plt.axvspan(xmin = 0, xmax= len([i for i in x_symb if i[2] == 'P']), alpha = 0.5, color = 'C0')
    # plt.axvspan(xmin = len([i for i in x_symb if i[2] == 'P'])+1,
    #             xmax= len([i for i in x_symb if i[2] in ['P','D']]),
    #             alpha = 0.5, color = 'C1')
    # plt.axvspan(xmin = len([i for i in x_symb if i[2] in ['P','D']])+1,
    #             xmax= len([i for i in x_symb if i[0] == 'I']),
    #             alpha = 0.5, color = 'C3')
    # plt.axvspan(xmin = len([i for i in x_symb if i[0] == 'I'])+1,
    #             xmax= len(x_symb),
    #             alpha = 0.5, color = 'C4')
    # plt.legend([r'Inventory $i\in \mathcal{P}$',
    #             r'Inventory $i\in \mathcal{D}$',
    #             r'Inventory $i\in \mathcal{R}$',
    #             r'Backorder $i\in \mathcal{R}$'])
    # plt.spy(eig_A*I - A)
    # plt.title(r'Rank $\lambda I-A$: '+ str(np.linalg.matrix_rank(eig_A*I - A)))
    # plt.show()

    fig, ax = plt.subplots(1, 1)# sharey=True, gridspec_kw={'width_ratios': [1, len(u_symb) / len(x_symb)]})
    # ax.figure(figsize=size)
    fig.set_figheight(2)
    fig.set_figwidth(8)
    C_mat = control.ctrb(A,B)
    ax.spy(C_mat, markersize=4, color='k')
    # ax.set_yticks([i + 0.05 for i in range(0,10,1)])
    # ax.set_xticks([i-0.05 for i in range(11,112,12)])
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        direction="in")

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        right=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelright=False,
        labelleft=False,
        direction="in")

    plt.tight_layout()
    plt.draw()
    plt.show()


    fig, axs = plt.subplots(1,2, sharey = True, gridspec_kw={'width_ratios': [1, len(u_symb)/len(x_symb)]})

    ax = axs[0]
    ax.spy(C, markersize=4, color='k')
    ax.axhline(y=-0.5, color='lightgrey', linestyle='-', linewidth=3)
    ax.axvline(x=-0.5, color='lightgrey', linestyle='-', linewidth=3)
    for i in range(1, 11):
        if i >= 3:
            ax.axhline(y=-0.5 + i, color='lightgrey', linestyle='-', linewidth=1)
            if i % 3 == 0:
                ax.axvline(x=-0.5 + i, color='lightgrey', linestyle='-', linewidth=1)
            else:
                ax.axvline(x=-0.5 + i, color='lightgrey', linestyle='--', linewidth=0.75)
        else:
            ax.axvline(x=-0.5 + i, color='lightgrey', linestyle='--', linewidth=0.75)
            ax.axhline(y=-0.5 + i, color='lightgrey', linestyle='--', linewidth=0.75)

    ax.axvline(x=-0.5 + i, color='lightgrey', linestyle='-', linewidth=3)
    ax.axhline(y=-0.5 + 6, color='lightgrey', linestyle='-', linewidth=3)
    ax.axis('off')

    axs[1].axis('off')

    plt.tight_layout()
    plt.draw()
    plt.show()

    fig, ax = plt.subplots(1, 1)  # sharey=True, gridspec_kw={'width_ratios': [1, len(u_symb) / len(x_symb)]})
    # ax.figure(figsize=size)
    fig.set_figheight(2)
    fig.set_figwidth(4)
    O_mat = control.obsv(A, C)
    ax.spy(np.transpose(O_mat), markersize=4, color='k')
    # ax.set_yticks([i + 0.05 for i in range(0, 10, 1)])
    # ax.set_xticks([i - 0.05 for i in range(11, 112, 12)])
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        direction="in")

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        right=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelright=False,
        labelleft=False,
        direction="in")

    plt.tight_layout()
    plt.draw()
    plt.show()


