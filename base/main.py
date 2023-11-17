import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from model import model
import model_MHE
import optimal_shipment_policy
from pyomo.util.infeasible import log_infeasible_constraints
import pickle as pk
import pyomo.environ as pyo
from instance_data import data_fun
# from instance_data_runtime_exp import data_fun  # Load this one for much larger supply chain network
from plotting import plotting_fun
from utils import q_sample, q_sample_scalar, implemented_policy_ship, implemented_policy_sales, estimator_round
import time
from pykalman import KalmanFilter
import os


# For results saving purposes
date = '0919'
path = 'results_'+date
if not os.path.exists(path):
    os.makedirs(path)

save_file = False
save_past = False
time_tracking = []
x_past_instance = {}
u_past_instance = {}
x_past_true_instance = {}
u_past_true_instance = {}
seed = -1
costs_dict = {}
demand_dict = {}
instances = ['perfect_fb', 'no_fb', 'no_fb_inv', 'estimation_fb']
instance_name = instances[3]

# Number of steps to run for the online simulation
K_steps = 10
# Number of instances to run
demand_instances = 10



for demand_instance in range(demand_instances):

    with open('./data/w_mean.pickle', 'rb') as handle:
        w_mean = pk.load(handle)
    with open('./data/w_cov.pickle', 'rb') as handle:
        w_cov = pk.load(handle)

    data = data_fun(20)
    w_mean = [0 for _ in data['R_set']]
    w_cov = [20 for _ in data['R_set']]
    # Initializing states and past decisions
    x_past = {'inventory': {}, 'backorder': {}}
    x_estimates = {'inventory': {}}
    u_past = {'shipment': {}, 'production': {}, 'sales': {}, 'waste': {}}

    x_past_true = {'inventory': {}, 'backorder': {}}
    u_past_true = {'shipment': {}, 'production': {}, 'sales': {}, 'waste': {}}

    demand_past = {}

    for i in data['R_set'] + data['D_set'] + data['P_set']:
        for q in range(1, data['quality'] + 1):
            for k in range(1, data['temperature'] + 1):
                if q == 30 and i[0] == 'R' and k == 1:
                    x_past['inventory'][(i, q, k, 0)] = 350
                    x_past_true['inventory'][(i, q, k, 0)] = 350
                    x_estimates['inventory'][(i, q, k, 0)] = 350
                else:
                    x_past['inventory'][(i, q, k, 0)] = 0
                    x_past_true['inventory'][(i, q, k, 0)] = 0
                    x_estimates['inventory'][(i, q, k, 0)] = 0

    for i in data['R_set']:
        x_past['backorder'][(i, 0)] = 0
        x_past_true['backorder'][(i, 0)] = 0

    q_levels = 30
    I = np.eye(q_levels - 1)
    A = np.hstack((np.zeros((q_levels - 1, 1)), I))
    A = np.vstack((A, np.zeros((1, q_levels))))
    C = np.ones((1, q_levels))

    # Running loop (this correspond to online simulation)
    seed += 101
    inputs = {}
    states = {}
    outputs = {}
    costs_inv = []
    costs_bo = []
    costs_prod = []
    costs_waste = []
    costs_ship = []
    sales_profit = []
    costs_all = []
    x_estimated = []
    x_true = []
    y_measurements = []

    solver_options = {'name': 'cplex', 'mipgap': 0.1e-2, 'timelimit': 5000}
    solver = pyo.SolverFactory(solver_options['name'])
    solver.options['mipgap'] = solver_options['mipgap']
    solver.options['timelimit'] = solver_options['timelimit']

    t0 = time.time()
    for t in range(K_steps):

        # Supply chain instance information
        data = data_fun(20 + t)
        for i in demand_past.keys():
            data['demand'][i[0]][i[1]] = demand_past[i]

        # Initializing model predictive control objects
        data['prod_distribution_nominal'] = {10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:1, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0}
        data_nominal = data.copy()
        data_nominal['prod_distribution'] = {13:0.05, 14:0.09, 15:0.19, 16:0.29, 17:0.19, 18:0.09, 19:0.05}
        data_nominal['prod_distribution'] = {10: 0.02, 11: 0.05, 12: 0.08, 13:0.08, 14:0.12, 15:0.15, 16:0.2, 17:0.15, 18:0.12,
                                             19:0.08, 20: 0.08, 21: 0.05, 22: 0.02}
        sum_values = sum([item for key, item in data_nominal['prod_distribution'].items()])
        data_nominal['prod_distribution'] = {key: item/sum_values for key, item in data_nominal['prod_distribution'].items() }
        m = model(data_nominal, x_past, u_past, t)

        data_true = data_nominal.copy()
        for key, value in data_nominal['prod_distribution'].items():
            np.random.seed(seed+key)
            data_true['prod_distribution'][key] = np.random.normal(value, value/2,size=(1,1))[0]
        sum_values = 1.1*sum([j for _,j in data_true['prod_distribution'].items()])[0]
        data_true['prod_distribution'] = {key: value/sum_values for key, value in data_true['prod_distribution'].items()}

        m_true = model(data_true, x_past_true, u_past_true, t)


        # Solving the model
        print('\n\n', '-------------- CONTROL t=%i, d_instance=%i --------------' % (t, demand_instance), '\n\n')
        results = solver.solve(m, tee=True, symbolic_solver_labels=True)
        results = solver.solve(m_true, tee=True, symbolic_solver_labels=True)



        # Computing (random) implementable actions. Dictionaries have keys (i,j,k,q) and (i,q) for shipment and sales.
        x_past_true_inventory = {}
        for i in m_true.N:
            for q in m_true.q:
                for k in m_true.k:
                    x_past_true_inventory[(i,q,k)] = np.round(m_true.I_iqkt[i,q,k,t].value,0)
        u_past_shipment = {}
        for i,j in m.A:
            for q in m.q:
                for k in m.k:
                    u_past_shipment[(i,j,q,k)] = np.round(m.x_ijqkt[i,j,q,k,t].value,0)
        m_policy = optimal_shipment_policy.policy(data, x_past_true_inventory, u_past_shipment)
        print('\n\n', '-------------- Shipment policy t=%i, d_instance=%i --------------' % (t, demand_instance), '\n\n')
        results = solver.solve(m_policy, tee=True, symbolic_solver_labels=True)
        # print('Auxiliary variabble value :', np.round(sum(m_policy.aux_var1[i,j].value for i,j in m_policy.A),2))


        # Computing (random) implementable actions. Dictionaries have keys (i,j,k,q) and (i,q) for shipment and sales.
        true_policy_shipment = implemented_policy_ship(m, m_true, data, t)


        # Extracting current decisions
        u0 = {'y': {}, 'x': {}, 'r': {}}
        x0 = {'inventory': {}, 'backorder': {}}
        y0 = {'omega': {}}
        rounding_policy = {}
        for i, j in data['A_routes']:
            for q in range(1, data['quality'] + 1):
                if (i,q) not in rounding_policy.keys():
                    rounding_policy[(i,q)] = 0

                for k in range(1, data['temperature'] + 1):
                    if instance_name == 'perfect_fb':
                    # Perfect feedback case
                        u_past['shipment'][(i, j, q, k, t)] = np.floor(max(0, m_true.x_ijqkt[i, j, q, k, t].value))
                        u_past_true['shipment'][(i, j, q, k, t)] = np.floor(max(0, m_true.x_ijqkt[i, j, q, k, t].value))
                    elif instance_name in  ['estimation_fb']:
                        # Random product quality shipment policy
                        u_past['shipment'][(i, j, q, k, t)] = np.floor(max(0, m.x_ijqkt[i, j, q, k, t].value))
                        u_past_true['shipment'][(i, j, q, k, t)] = np.floor(m_policy.x_ijqk[i,j,q,k].value-rounding_policy[(i,q)]) #
                        rounding_policy[(i, q)] = u_past_true['shipment'][(i, j, q, k, t)] - (m_policy.x_ijqk[i,j,q,k].value-rounding_policy[(i,q)])
                    elif instance_name in ['no_fb', 'no_fb_inv']:
                        # Random product quality shipment policy
                        u_past['shipment'][(i, j, q, k, t)] = np.floor(max(0, m.x_ijqkt[i, j, q, k, t].value))
                        #np.round(max(0,true_policy_shipment[(i, j, q, k)]),0)
                        u_past_true['shipment'][(i, j, q, k, t)] = np.floor(m_policy.x_ijqk[i, j, q, k].value - rounding_policy[(i, q)])  #
                        rounding_policy[(i, q)] = u_past_true['shipment'][(i, j, q, k, t)] - (m_policy.x_ijqk[i, j, q, k].value - rounding_policy[(i, q)])

                    u0['x'][(i, j, q, k)] = np.floor(max(0, u_past_true['shipment'][(i, j, q, k, t)]))



        print(' ')
        for i in data['P_set']:
            for q in range(1, data['quality'] + 1):

                if instance_name == 'perfect_fb':
                    # Perfect feedback case
                    u_past['production'][(i, q, t)] = round(max(0, m_true.y_iqt[i, q, t].value), 0)
                    u_past_true['production'][(i, q, t)] = round(max(0, m_true.y_iqt[i, q, t].value),0)

                elif instance_name in ['no_fb', 'no_fb_inv','estimation_fb']:
                    # Imperfect feedback (controller implementation)
                    u_past['production'][(i, q, t)] = round(max(0, m.y_iqt[i, q, t].value), 0)
                    u_past_true['production'][(i, q, t)] = round(max(0, m_true.y_iqt[i, q, t].value),0)

                u0['y'][(i, q)] = round(max(0, u_past_true['production'][(i, q, t)]), 0)



        # Simulating demand realization with random seed for reproducibility
        noise_sample = multivariate_normal(w_mean, np.array([i for i in w_cov]), seed = seed+t).rvs(1)

        # print(noise_sample)

        # Estimated model
        m = model(data_nominal, x_past, u_past, t)
        m.prod_quality.deactivate()
        j = 0
        for i in m.R:
            demand_past[(i, t)] = m.d_it[i, t].value + np.round(noise_sample[j], 0)
            m.d_it[i, t] = m.d_it[i, t].value + np.round(noise_sample[j], 0)
            j += 1

        # Solving the model
        print('\n\n', '-------------- SIMULATION (Estimated) t=%i --------------' % t, '\n\n')
        results = solver.solve(m, tee=True, symbolic_solver_labels=True)

        # True model
        m_true = model(data_true, x_past_true, u_past_true, t)
        m_true.prod_quality.deactivate()
        j = 0
        for i in m.R:
            demand_past[(i, t)] = m_true.d_it[i, t].value + np.round(noise_sample[j], 0)
            m_true.d_it[i, t] = m_true.d_it[i, t].value + np.round(noise_sample[j], 0)
            j += 1

        # Solving the model
        print('\n\n', '-------------- SIMULATION (True) t=%i --------------' % t, '\n\n')
        results = solver.solve(m_true, tee=True, symbolic_solver_labels=True)

        # m_true.c10_1.pprint()

        costs_t0 = m_true.prod_costs[t].value + m_true.ship_costs[t].value + m_true.waste_costs[t].value + \
                   m_true.inv_costs[t].value + m_true.backorder_costs[t].value - (1/400)*m_true.sales_profit[t].value

        # Extracting current decisions
        for i in data['R_set']:
            for q in range(1, data['quality'] + 1):
                u_past['sales'][(i, q, t)] = round(max(0, m.s_iqt[i, q, t].value), 0)
                u_past_true['sales'][(i, q, t)] = round(max(0, m_true.s_iqt[i, q, t].value), 0)
                u0['r'][(i, q)] = np.round(max(0, m_true.s_iqt[i, q, t].value), 0)
        for i in data['R_set'] + data['D_set'] + data['P_set']:
            u_past['waste'][(i, t)] = np.round(max(0, m.omega_it[i, t].value), 0)
            u_past_true['waste'][(i, t)] = np.round(max(0, m_true.omega_it[i, t].value), 0)
            y0['omega'][i] = np.round(max(0, m_true.omega_it[i, t].value), 0)



        # Simulating quality distribution uncertainty
        for i in data['R_set'] + data['D_set'] + data['P_set']:
            for k in range(1, data['temperature'] + 1):
                q_dist_prev = 0.0
                for q in range(1, data['quality'] + 1):

                    if q == 1:
                        q_dist = q_sample_scalar([max(np.round(m_true.I_iqkt[i, q, k, t + 1].value, 0), 0),
                                                    max(np.round(m_true.I_iqkt[i, q + 1, k, t + 1].value, 0), 0)], seed = q*(seed+t)*int(i[1]))
                    elif q == data['quality']:
                        q_dist = 0
                    else:
                        q_dist = q_sample_scalar([max(np.round(m_true.I_iqkt[i, q, k, t + 1].value-q_dist_prev, 0), 0),
                                                      max(np.round(m_true.I_iqkt[i, q + 1, k, t + 1].value, 0), 0)], seed = q*(seed+t)*int(i[1]))


                    # Feedback without using estimation
                    if instance_name in ['no_fb', 'no_fb_inv']:
                        x_past['inventory'][(i, q, k, t + 1)] = np.round(max(0, m.I_iqkt[i, q, k, t + 1].value),0)
                    # Perfect feedback
                    elif instance_name in ['perfect_fb']:
                        x_past['inventory'][(i, q, k, t + 1)] = np.round(max(0, m_true.I_iqkt[i, q, k, t + 1].value - q_dist_prev + q_dist),0)

                    x_past_true['inventory'][(i, q, k, t + 1)] = np.round(max(0, m_true.I_iqkt[i, q, k, t + 1].value) - q_dist_prev + q_dist,0)
                    x0['inventory'][(i, q, k)] = np.round(max(0, x_past_true['inventory'][(i, q, k, t + 1)]),0)

                    q_dist_prev = q_dist

        # Performing MHE
        if instance_name in ['estimation_fb']:
            data_estimation = data_fun(t + 1)
            m_estimate = model_MHE.perform_estimate(data_estimation, 0, u_past_true, x_past_true, x_estimates)

            for i in data['R_set'] + data['D_set'] + data['P_set']:
                for k in range(1, data['temperature'] + 1):
                    for q in range(1, data['quality'] + 1):
                        x_past['inventory'][(i, q, k, t + 1)] = np.round(max(0, m_estimate.I_iqkt[i, q, k, t + 1].value), 0)
                        x_estimates['inventory'][(i, q, k, t + 1)] = np.round(max(0, m_estimate.I_iqkt[i, q, k, t + 1].value), 0)



        for i in data['R_set']:

            if instance_name in ['perfect_fb', 'no_fb_inv', 'estimation_fb']:
                # Feedback when measuring backorder
                x_past['backorder'][(i, t + 1)] = np.round(max(0, m_true.BO_it[i, t + 1].value),0)
            elif instance_name == 'no_fb':
                # No feedback on backorder employed
                x_past['backorder'][(i, t + 1)] = np.round(max(0, m.BO_it[i, t + 1].value), 0)

            x_past_true['backorder'][(i, t + 1)] = np.round(max(0, m_true.BO_it[i, t + 1].value),0)
            x0['backorder'][i] = np.round(max(0, m_true.BO_it[i, t + 1].value),0)

        costs_all.append(costs_t0)
        costs_inv.append(m_true.inv_costs[t].value)
        costs_bo.append(m_true.backorder_costs[t].value)
        costs_prod.append(m_true.prod_costs[t].value)
        costs_waste.append(m_true.waste_costs[t].value)
        costs_ship.append(m_true.ship_costs[t].value)
        sales_profit.append(m_true.sales_profit[t].value)

        inputs[t] = u0
        states[t] = x0
        outputs[t] = y0

    t1 = time.time()
    time_tracking.append(t1-t0)
    costs_dict[demand_instance] = {'all': costs_all, 'inventory': costs_inv, 'backorder': costs_bo, 'production': costs_prod, 'waste': costs_waste,
                                   'shipment': costs_ship, 'sales_profit': sales_profit}
    demand_dict[demand_instance] = data['demand']

    x_past_instance[demand_instance] = x_past
    x_past_true_instance[demand_instance] = x_past_true
    u_past_instance[demand_instance] = u_past
    u_past_true_instance[demand_instance] = u_past_true



if save_file:
    with open('./results2_'+date+'/costs_'+instance_name+'.pickle', 'wb') as handle:
        pk.dump(costs_dict, handle, protocol=pk.HIGHEST_PROTOCOL)

    with open('./results2_'+date+'/demand_instances.pickle', 'wb') as handle:
        pk.dump(demand_dict, handle, protocol=pk.HIGHEST_PROTOCOL)

if save_past:

    with open('./data_estimation2_'+date+'/'+instance_name+'_demand_dict.pickle', 'wb') as handle:
        pk.dump(demand_dict, handle, protocol=pk.HIGHEST_PROTOCOL)

    with open('./data_estimation2_'+date+'/'+instance_name+'_x_past.pickle', 'wb') as handle:
        pk.dump(x_past_instance, handle, protocol=pk.HIGHEST_PROTOCOL)

    with open('./data_estimation2_'+date+'/'+instance_name+'_x_past_true.pickle', 'wb') as handle:
        pk.dump(x_past_true_instance, handle, protocol=pk.HIGHEST_PROTOCOL)

    with open('./data_estimation2_'+date+'/'+instance_name+'_u_past.pickle', 'wb') as handle:
        pk.dump(u_past_instance, handle, protocol=pk.HIGHEST_PROTOCOL)

    with open('./data_estimation2_'+date+'/'+instance_name+'_u_past_true.pickle', 'wb') as handle:
        pk.dump(u_past_true_instance, handle, protocol=pk.HIGHEST_PROTOCOL)

#
# if save_file:
#     path_to_save = './figures/'+instance_name+'/'
#     plotting_fun([], [], [], data, ['costs'], costs_all); plt.draw() ; plt.savefig(path_to_save+'costs'); plt.close()
#     plotting_fun(states, inputs, outputs, data, ['BO']); plt.draw() ; plt.savefig(path_to_save+'backorder') ; plt.close()
#     plotting_fun(states, inputs, outputs, data, ['omega']) ; plt.draw() ; plt.savefig(path_to_save+'waste') ; plt.close()
#     plotting_fun(states, inputs, outputs, data, ['s', data['R_set']], demand=demand_past); plt.draw() ; plt.savefig(path_to_save+'sales') ; plt.close()
#     plotting_fun(states, inputs, outputs, data, ['I', data['R_set'] + data['D_set'] + data['P_set']]); plt.draw() ; plt.savefig(path_to_save+'inventory') ; plt.close()
#     plotting_fun(states, inputs, outputs, data, ['x', data['A_routes']]); plt.draw() ; plt.savefig(path_to_save+'shipment') ; plt.close()

# plt.show()
#
# plt.figure(figsize=(3, 2))
# line_true = plt.plot(x_true, color='b')
# line_estimate = plt.plot(x_estimated, color='r', linestyle='--')
# plt.legend((line_true[0], line_estimate[0]), ('True', 'Estimated'), loc='upper right')
# plt.xlabel('Time (days)');
# plt.ylabel(r'Inventories $I_q$');
# plt.title(r'$\hat{\bf{x}}$', weight='bold')
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(3, 2))
# plt.plot(y_measurements)
# plt.xlabel('Time (days)')
# plt.ylabel('Net inventory')
# plt.title('y', weight='bold')
# plt.axhline(y=0, linestyle='--', color='k', linewidth=1)
# plt.tight_layout()
# plt.show()

# sum(m_policy.x_ijqk['D1',j,14,1,5].value for j in m_policy.n_i['D1'])
# sum(u_past_true['shipment'][('D1',j,19,1,4)] for j in m_policy.n_i['D1'])
# x_past_true['inventory'][('D1',20,1,4)]
#
# for j in m_policy.n_i['D1'] :
#     print(u_past_true['shipment'][('D1', j, 14, 1, 5)], np.round(max(0, m_policy.x_ijqk['D1', j, 14, 1].value), 0), m_policy.x_ijqk['D1', j, 14, 1].value, x_past_true['inventory'][('D1', 15, 1, 5)])
