import pyomo.environ as pyo
import numpy as np




def model(data, t_current, measurements):
    """
    Function that defines the Pyomo model instance for the supply chain network data for STATE ESTIMATION.

      Inputs:
      - data: dict: Corresponding to all the supply chain network parameters (e.g., facilities, network connection,
      cost coefficients, lead times, etc.)
      - t_current: int: Corresponds to the present time at which the simulation is run
      - measurements: dict: Dictionary of past measurements of supply chain output variables

      Outputs:
      - m: pyo.ConcreteModel(): Pyomo instance of the supply chain quality estimation problem
    """

    m = pyo.ConcreteModel()

    # Quality index
    m.q = pyo.RangeSet(1, data['quality'])
    # Temperature index
    m.k = pyo.RangeSet(1, data['temperature'])
    # Optimization horizon
    m.H = pyo.Param(initialize=data['H'], mutable=True)
    # Time grid
    m.t_x = pyo.RangeSet(0, data['H'])
    m.t_u = pyo.RangeSet(0, data['H'] - 1)

    # Set of production sites
    m.P = pyo.Set(initialize=data['P_set'])
    # Set of distribution centers
    m.D = pyo.Set(initialize=data['D_set'])
    # Set of retailers
    m.R = pyo.Set(initialize=data['R_set'])
    # Set of all facilities
    m.N = (m.P | m.D | m.R)
    # Set of all routes
    m.A = pyo.Set(dimen=2, initialize=data['A_routes'])

    # Set of all successors to node i
    m.n_i = pyo.Set(m.N, initialize=data['succ'])
    m.v_i = pyo.Set(m.N, initialize=data['pred'])

    # Large positive value (used in Big-M constraint)
    m.M = pyo.Param(initialize=1e6)

    # Shipment lead times
    def u_ij_init(m, i, j):
        return data['ship_lead_time'][(i, j)]

    m.u_ij = pyo.Param(m.A, initialize=u_ij_init)

    # Demand trajectory
    def d_it_init(m, i, t):
        return measurements['demand'][i][t]

    m.d_it = pyo.Param(m.R, m.t_u, initialize=d_it_init, mutable=True)

    # Production capacity
    def _p_UB_init(m, i, t):
        return data['production_UB'][i][t]

    m.p_UB = pyo.Param(m.P, m.t_u, initialize=_p_UB_init)


    # Inventory with quality level q in facility i with temperature level k at beginning of period t
    m.I_iqkt = pyo.Var(m.N, m.q, m.k, m.t_x, within=pyo.NonNegativeReals)

    # Flow quantities on arcs (i,j) in period t, temperature level k with quality q starting from node i
    m.x_ijqkt = pyo.Var(m.A, m.q, m.k, m.t_u, within=pyo.NonNegativeReals, initialize=0)

    # Number of batches with quality level q at the end of the period t required to be produced in plant i
    m.y_iqt = pyo.Var(m.P, m.q, m.t_u, within=pyo.NonNegativeReals)

    # Amount of waste in facility i in period t
    m.omega_it = pyo.Var(m.N, m.t_u, within=pyo.NonNegativeReals)

    # Amount sold in facility i at quality level 1 at in period t
    m.s_iqt = pyo.Var(m.R, m.q, m.t_u, within=pyo.NonNegativeReals)

    # Missed sales variable
    m.BO_it = pyo.Var(m.R, m.t_x, within=pyo.NonNegativeReals)

    # Quality disturbance
    m.eps_iqq = pyo.Var(m.N, m.q, m.q, m.t_x,  within=pyo.Reals, initialize=0)
    for i in m.N:
        for t in m.t_x:
            for q1 in m.q:
                for q2 in m.q:
                    if (q1+1) == q2 or (q1-1) == q2:
                        pass
                    else:
                        m.eps_iqq[i,q1,q2, t] = 0
                        m.eps_iqq[i, q1, q2, t].fixed = True

    def eps_rule1(m,i,q, t):
        if q-1>= 1:
            return m.eps_iqq[i,q,q-1,t] == -m.eps_iqq[i,q-1,q,t]
        else:
            return pyo.Constraint.Skip
    m.c_eps1 = pyo.Constraint(m.N, m.q, m.t_x, rule = eps_rule1)

    def eps_rule2(m,i,q, t):
        if q+1 <= data['maximum_quality'][i]:
            return m.eps_iqq[i,q,q+1,t] == -m.eps_iqq[i,q+1,q,t]
        else:
            return pyo.Constraint.Skip
    m.c_eps2 = pyo.Constraint(m.N, m.q, m.t_x, rule = eps_rule2)

    # def eps_rule3(m, i, q1, q2, t):
    #     if q1 == q2:
    #         return pyo.Constraint.Skip
    #     else:
    #         return m.eps_iqq[i,q1, q2,t] >= -sum(m.I_iqkt[i,q1,k,t] for k in m.k)
    # m.c_eps3 = pyo.Constraint(m.N, m.q, m.q,  m.t_x, rule = eps_rule3)
    #
    # def eps_rule4(m, i, q1, q2, t):
    #     if q1 == q2:
    #         return pyo.Constraint.Skip
    #     else:
    #         return m.eps_iqq[i,q1, q2,t] <= sum(m.I_iqkt[i,q2,k,t] for k in m.k)
    # m.c_eps4 = pyo.Constraint(m.N, m.q, m.q,  m.t_x, rule = eps_rule4)

    # m.eps_iqt = pyo.Var(m.N, m.q, m.t_x, within=pyo.Reals, initialize = 0)


    # def _eps_rule(m,i,t):
    #     return sum(m.eps_iqt[i, q, t] for q in m.q) == 0
    # m.eps_rule = pyo.Constraint(m.N, m.t_x, rule=_eps_rule)


    # Fixing that nothing can be sold bellow the minimum quality required
    # for i in m.R:
    #     for q in m.q:
    #         if q <= data['minimum_quality'][i]:
    #             for t in m.t_u:
    #                 m.s_iqt[i, q, t] = 0
    #                 m.s_iqt[i, q, t].fixed = True

    # Fixing that nothing arriving at a quality less than requires to zero
    # for i, j in m.A:
    #     for q in m.q:
    #         for k in m.k:
    #             for t in m.t_u:
    #                 if q < data['minimum_quality'][i] + data['degradation_ship'][(i, j)][k]:
    #                     m.x_ijqkt[i, j, q, k, t] = 0
    #                     m.x_ijqkt[i, j, q, k, t].fixed = True

    # Fixing that nothing can be produced with quality less than minimum quality at P
    # for i in m.P:
    #     for q in m.q:
    #         # if q < data['minimum_quality'][i]:
    #         if q != 10:
    #             for t in m.t_u:
    #                 m.y_iqt[i, q, t] = 0
    #                 m.y_iqt[i, q, t].fixed = True

    # Fixing that no inventory is stored under the minimum quality
    # for i in m.N:
    #     for q in m.q:
    #         if q < data['minimum_quality'][i]:
    #             for k in m.k:
    #                 for t in m.t_x:
    #                     m.I_iqkt[i, q, k, t] = 0
    #                     m.I_iqkt[i, q, k, t].fixed = True

    if t_current <= 20:
        prev_estimates = 1
    else:
        prev_estimates = 1

    # Objective function: minimize total costs        # sum(sum(sum((1/7)*(m.eps_iqt[i, q, t])**2 for i in m.N) for q in m.q) for t in m.t_u)
    def _obj_rule(m):
        return (1/5)*sum(sum(sum(sum((m.eps_iqq[i, q1, q2, t])**2 for i in m.N) for q1 in m.q) for q2 in m.q) for t in m.t_x)+\
               prev_estimates*sum((m.I_iqkt[i[0],i[1],i[2],i[3]] - value)**2 for i, value in measurements['previous_estimate'].items()) +\
               sum(m.BO_it[i, 0]**2 for i in m.R)

    m.obj = pyo.Objective(rule = _obj_rule,
                          sense=pyo.minimize)

    # Measuring production (or measuring inventory at the producer)
    for i in m.P:
        for q in m.q:
            for t in m.t_u:
                m.y_iqt[i,q,t] = measurements['production'][(i, q, t)]
                m.y_iqt[i, q, t].fixed = True
    for i in m.R:
        for q in m.q:
            for t in m.t_u:
                m.s_iqt[i, q, t] = measurements['sales'][(i, q, t)]
                m.s_iqt[i, q, t].fixed = True
    for i in m.N:
        for t in m.t_u:
            m.omega_it[i, t] = measurements['waste'][(i, t)]
            m.omega_it[i, t].fixed = True

    # Sporadic quality measurements
    def _quality_measurements_rule(m, i, q, k, t):
        if i[0] == 'D':
            if (i,q,k,t) in measurements['inventory_q'].keys():
                return m.I_iqkt[i, q, k, t] >= measurements['inventory_q'][(i,q,k,t)]
            else:
                return pyo.Constraint.Skip
        if i[0] == 'P' and (i, q, k, t) in measurements['inventory_q'].keys():
            return m.I_iqkt[i, q, k, t] == measurements['inventory_q'][(i, q, k, t)]
        else:
            return pyo.Constraint.Skip

    m.quality_measurements = pyo.Constraint(m.N, m.q, m.k, m.t_u, rule = _quality_measurements_rule)

    # for i in m.D:
    #     for q in m.q:
    #         for k in m.k:
    #             for t in m.t_u:
    #                 if (i,q,k,t) in measurements['inventory_q'].keys():
    #                     m.I_iqkt[i, q, k, t] = measurements['inventory_q'][(i,q,k,t)]
    #                     m.I_iqkt[i, q, k, t].fixed = True



    def _net_inventory_rule(m, i, t):
        return sum(m.I_iqkt[i,q,1,t]for q in m.q) == measurements['net_inventory'][(i,t)]

    m.net_inventory_rule = pyo.Constraint(m.N, m.t_x, rule = _net_inventory_rule)

    def _net_shipment_rule(m, i, j, t):
        return sum(m.x_ijqkt[i,j,q,1,t] for q in m.q) == measurements['net_shipment'][(i,j,t)]

    m.net_shipment_rule = pyo.Constraint(m.A, m.t_u, rule = _net_shipment_rule)



    # Inventory balance at the producer
    def c9_rule1(m, i, q, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            if q >= data['minimum_quality'][i] and q <= data['maximum_quality'][i] and t + 1 <= pyo.value(m.H):

                return sum(m.I_iqkt[i, q, k, t + 1] for k in m.k) == \
                       (sum(m.I_iqkt[i, q + data['degradation'][k], k, t] for k in m.k if
                            (q + data['degradation'][k] <= data['maximum_quality'][i]))) + \
                       sum([m.y_iqt[i, q, t - data['prod_lead_time'][i]] if t - data['prod_lead_time'][i] >= 0 else 0]) + \
                       - sum(sum(m.x_ijqkt[i, j, q, k, t] for j in m.n_i[i] if
                                 q >= data['minimum_quality'][j] + data['degradation_ship'][(i, j)][k])
                             for k in m.k) - sum([m.eps_iqq[i,q,q-1,t+1] if q-1 >=1 else 0] + [m.eps_iqq[i,q,q+1,t+1] if q + 1 <= data['maximum_quality'][i] else 0])
                        # m.eps_iqt[i,q,t+1]
            else:
                return pyo.Constraint.Skip

    m.c9_1 = pyo.Constraint(m.P, m.q, m.t_u, rule=c9_rule1)


    # Inventory balance at the distribution center
    def c10_rule1(m, i, q, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            if q >= data['minimum_quality'][i] and q <= data['maximum_quality'][i] and t + 1 <= pyo.value(m.H):
                return sum(m.I_iqkt[i, q, k, t + 1] for k in m.k) == \
                       (sum(m.I_iqkt[i, q + data['degradation'][k], k, t] for k in m.k if
                            (q + data['degradation'][k] <= data['maximum_quality'][i]))) + \
                       sum(sum(m.x_ijqkt[j, i, q + data['degradation_ship'][(j, i)][k], k, t - m.u_ij[j, i]]
                               for j in m.v_i[i] if t - m.u_ij[j, i] >= 0 if q + data['degradation_ship'][(j, i)][k] <=
                               data['maximum_quality'][i]) for k in m.k) - \
                       sum(sum(m.x_ijqkt[i, j, q, k, t] for j in m.n_i[i] if
                               q >= data['minimum_quality'][j] + data['degradation_ship'][(i, j)][k]) for k in m.k) -\
                       sum([m.eps_iqq[i,q,q-1,t+1] if q-1 >=1 else 0] + [m.eps_iqq[i,q,q+1,t+1] if q + 1 <= data['maximum_quality'][i] else 0])
            else:
                return pyo.Constraint.Skip

    m.c10_1 = pyo.Constraint(m.D, m.q, m.t_u, rule=c10_rule1)

    # Inventory balance at the retailer
    def c10R_rule1(m, i, q, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            if q >= data['minimum_quality'][i] and q <= data['maximum_quality'][i] and t + 1 <= pyo.value(m.H):
                return sum(m.I_iqkt[i, q, k, t + 1] for k in m.k) == \
                       (sum(m.I_iqkt[i, q + data['degradation'][k], k, t] for k in m.k if
                            (q + data['degradation'][k] <= data['maximum_quality'][i]))) + \
                       sum(sum(m.x_ijqkt[j, i, q + data['degradation_ship'][(j, i)][k], k, t - m.u_ij[j, i]]
                               for j in m.v_i[i] if t - m.u_ij[j, i] >= 0 if
                               q + data['degradation_ship'][(j, i)][k] <= data['maximum_quality'][i]) for k in m.k) -\
                       sum([m.eps_iqq[i,q,q-1,t+1] if q-1 >=1 else 0] + [m.eps_iqq[i,q,q+1,t+1] if q + 1 <= data['maximum_quality'][i] else 0]) - m.s_iqt[i, q, t]
            else:
                return pyo.Constraint.Skip

    m.c10R_1 = pyo.Constraint(m.R, m.q, m.t_u, rule=c10R_rule1)


    # Backorder balance at retailers
    def c10R_rule2(m, i, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            if t + 1 <= pyo.value(m.H):
                return m.BO_it[i, t + 1] == m.BO_it[i, t] + m.d_it[i, t] - sum(
                    m.s_iqt[i, q, t] for q in m.q if q >= data['minimum_quality'][i])
            else:
                return pyo.Constraint.Skip

    m.c10R_2 = pyo.Constraint(m.R, m.t_u, rule=c10R_rule2)

    # m.c10R_2.display()

    # Waste balance
    def c11_rule1(m, i, t):
        if t < t_current:
            return pyo.Constraint.Skip
        elif i[0] =='P':
            return m.omega_it[i, t] >= \
                   sum(sum(m.I_iqkt[i, q, k, t] for q in m.q if
                           q <= data['minimum_quality'][i] + data['degradation'][k] - 1) for k in m.k)
        elif i[0] in ['D','R']:
            return m.omega_it[i, t] >= \
                   sum(sum(m.I_iqkt[i, q, k, t] for q in m.q if
                           q <= data['minimum_quality'][i] + data['degradation'][k] - 1) for k in m.k) + \
                   sum(sum(sum(m.x_ijqkt[j, i, q + data['degradation_ship'][(j, i)][k],k, t - m.u_ij[j, i]]
                               for q in m.q if q <= data['minimum_quality'][i] + data['degradation_ship'][(j, i)][k] - 1)
                           for k in m.k) for j in m.v_i[i] if t - m.u_ij[j, i] >= 0)

    m.c11_1 = pyo.Constraint(m.N, m.t_u, rule=c11_rule1)

    # Sales must not exceed backorder plus demand
    def c12_rule1(m, i, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            return sum(m.s_iqt[i, q, t] for q in m.q if q >= data['minimum_quality'][i]) - m.BO_it[i, t] - m.d_it[i, t] <= 0

    m.c12_1 = pyo.Constraint(m.R, m.t_u, rule=c12_rule1)



    # Production capacity constraint
    def c18_rule1(m, i, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            return sum(m.y_iqt[i, q, t] for q in m.q if q >= data['minimum_quality'][i]) <= m.p_UB[i, t]

    m.c18_1 = pyo.Constraint(m.P, m.t_u, rule=c18_rule1)

    # Inventory must be greater than sales (for retailers)
    def c21_rule1(m, i, q, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            if q >= data['minimum_quality'][i] and q <= data['maximum_quality'][i]:
                return sum(m.I_iqkt[i, q + data['degradation'][k], k, t] for k in m.k if
                           q + data['degradation'][k] <= data['maximum_quality'][i]) >= m.s_iqt[i, q, t]
            else:
                return pyo.Constraint.Skip

    m.c21_1 = pyo.Constraint(m.R, m.q, m.t_u, rule=c21_rule1)

    # Inventory must be greater than shipment (for distribution centers and producers)
    def c22_rule(m, i, q, t):
        if t < t_current:
            return pyo.Constraint.Skip
        if i[0] == 'R':
            return pyo.Constraint.Skip
        else:
            #if q >= data['minimum_quality'][i] and q <= data['maximum_quality'][i]:
            return sum(m.I_iqkt[i, q + data['degradation'][k], k, t] for k in m.k if
                           q + data['degradation'][k] <= data['maximum_quality'][i])  >=  \
                   sum(sum(m.x_ijqkt[i,j,q,k,t] for j in m.n_i[i]) for k in m.k)
            # else:
            #     return pyo.Constraint.Skip

    m.c22 = pyo.Constraint(m.N, m.q, m.t_u, rule=c22_rule)

    return m

def perform_estimate(data, t_current, u_past_true,x_past_true, x_estimates):

    # Gathering measurements
    measurements = {'production': {}, 'sales': {}, 'waste': {}, 'net_inventory': {}, 'net_shipment': {},
                    'previous_estimate': {}, 'inventory_q': {}}

    for key, value in u_past_true['production'].items():
        if key[0][0] == 'P':
            measurements['production'][key] = value
    for key, value in u_past_true['sales'].items():
        measurements['sales'][key] = value
    for key, value in u_past_true['waste'].items():
        measurements['waste'][key] = value
    for key, value in x_past_true['inventory'].items():
        if (key[0], key[3]) not in measurements['net_inventory'].keys():
            measurements['net_inventory'][(key[0], key[3])] = value
        else:
            measurements['net_inventory'][(key[0], key[3])] += value
        if key[3] == 0:
            measurements['previous_estimate'][(key[0], key[1], key[2], key[3])] = value
    for key, value in u_past_true['shipment'].items():
        if (key[0], key[1], key[4]) not in measurements['net_shipment'].keys():
            measurements['net_shipment'][(key[0], key[1], key[4])] = value
        else:
            measurements['net_shipment'][(key[0], key[1], key[4])] += value
    measurements['demand'] = data['demand']
    percentage_inventory_measured = 0
    for key, value in x_past_true['inventory'].items():
        if key[0][0] == 'D' and value > 0:
            coin = np.random.choice([0, 1], 1, p=[1-percentage_inventory_measured, percentage_inventory_measured])[0]
            if coin == 1:
                measurements['inventory_q'][key[0], key[1], key[2], key[3]] = value
        if key[0][0] == 'P':
            measurements['inventory_q'][key[0], key[1], key[2], key[3]] = value
    for key, value in x_estimates['inventory'].items():
        measurements['previous_estimate'][key] = value


    solver_options = {'name': 'cplex', 'mipgap': 0.1e-3, 'timelimit': 5000}
    solver = pyo.SolverFactory(solver_options['name'])
    solver.options['mipgap'] = solver_options['mipgap']
    solver.options['timelimit'] = solver_options['timelimit']

    m = model(data, t_current, measurements)
    results = solver.solve(m, tee=True, symbolic_solver_labels=True)

    return m