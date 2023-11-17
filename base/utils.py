



def q_sample(q_range,means = []):

    dist_vec = []
    for q in range(0,q_range-1):

        import scipy.stats as ss
        import numpy as np

        x = np.arange(-means[q][0], means[q][1]+1)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=8) - ss.norm.cdf(xL, scale=8)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        nums = np.random.choice(x, size=1, p=prob)


        dist_vec.append(7*nums[0])

    return dist_vec

def q_sample_scalar(means, seed = None):
    import scipy.stats as ss
    import numpy as np


    if seed is None:
        pass
    else:
        np.random.seed(seed)

    x = np.arange(-means[0], means[1]+1)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=7) - ss.norm.cdf(xL, scale=7)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size=1, p=prob)

    return nums[0]



def implemented_policy_ship(m_control, m_true, data, t):
    import random

    quality_available = {}
    for i in data['P_set'] + data['D_set']:

        quality_available[i] = []

        if i[0] == "P":
            for q_vals in m_true.q:
                quality_available[i] += [q_vals for _ in  range(int(round(sum(m_true.I_iqkt[i, q_vals + data['degradation'][k], k, t].value
                                                for k in m_true.k if q_vals + data['degradation'][k] <= data['maximum_quality'][i]), 0)))]
                # quality_available[i] += [q_vals for _ in range(int(round(m_control.y_iqt[i, q_vals, t].value, 0)))]
        if i[0] == "D":
            for q_vals in m_true.q:
                quality_available[i] += [q_vals for _ in range(int(round(sum(m_true.I_iqkt[i, q_vals + data['degradation'][k], k, t].value
                                                for k in m_true.k if q_vals + data['degradation'][k] <= data['maximum_quality'][i]), 0)))]
                # quality_available[i] += [q_vals for _ in range(int(round(sum(sum(m_true.x_ijqkt[j, i, q_vals +
                #                                 data['degradation_ship'][(j, i)][k], k, t -  m_control.u_ij[j, i]].value
                #                                      for j in m_true.v_i[i] if t - m_true.u_ij[j, i] >= 0 if q_vals +
                #                                     data['degradation_ship'][(j, i)][k] <=  data['maximum_quality'][i])
                #                                     for k in m_true.k), 0)))]

    true_policy_shipment = {}


    for i,j in m_control.A:
        for k in m_control.k:
                if sum(m_control.x_ijqkt[i,j,q,k,t].value for q in m_control.q) > 0:

                    # Random sample of inventory quality (number of samples corresponds to amount shipped)5
                    q_shipped = random.sample(quality_available[i],
                                                k = int(min(round(sum(m_control.x_ijqkt[i,j,q,k,t].value for q in m_control.q),0),
                                                        len(quality_available[i]))))

                    for ind_1, val_1 in enumerate(q_shipped):
                        for ind_2, val_2 in enumerate(quality_available[i]):
                            if val_1 == val_2:
                                quality_available[i].pop(ind_2)
                                break

                    for q_vals in q_shipped:
                        if (i, j, q_vals, k) in true_policy_shipment.keys():
                            true_policy_shipment[(i, j, q_vals, k)] += 1
                        else:
                            true_policy_shipment[(i, j, q_vals, k)] = 1

    for i,j in m_control.A:
        for k in m_control.k:
            for q in m_control.q:
                if (i,j,q,k) not in true_policy_shipment.keys():
                    true_policy_shipment[(i,j,q,k)] = 0

    return true_policy_shipment


def implemented_policy_sales(m_control, m_true, data, t):
    import random

    true_policy_sales = {}
    for i in m_control.R:
        for q in m_control.q:

            if m_control.s_iqt[i, q, t].value > 0.1:

                quality_available = []
                for q_vals in m_true.q:
                    quality_available += [q_vals for _ in range(int(sum(m_true.I_iqkt[i, q_vals, k, t].value
                                                                        for k in m_true.k)))]

                # Random sample of inventory quality (number of samples corresponds to amount sold)
                q_sold = random.sample(quality_available,
                                          k=int(m_control.s_iqt[i, q, t].value))

                for q_vals in q_sold:
                    if (i, q_vals) in true_policy_sales.keys():
                        true_policy_sales[(i, q_vals)] += 1
                    else:
                        true_policy_sales[(i, q_vals)] = 1

    for i in m_control.R:
        for q in m_control.q:
            if (i,q) not in true_policy_sales.keys():
                true_policy_sales[(i,q)] = 0

    return true_policy_sales


def estimator_round(vec):

    surplus = 0.0
    max_val = [0.0,-float('inf')]
    for i,j in enumerate(vec):

        if j < 3:
            surplus += j
            vec[i] = 0.0

        if max_val[1] < j:
            max_val = [i,j]

        surplus += (vec[i] - round(vec[i],0))
        vec[i] = round(vec[i],0)

    vec[max_val[0]] += round(surplus,0)

    vec = [max(0.0,i) for i in vec]

    return vec







