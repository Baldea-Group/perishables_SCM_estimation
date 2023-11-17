import pyomo.environ as pyo
import numpy as np
import pickle as pk

with open('./data_estimation/x_past_true.pickle', 'rb') as handle:
    x_past_true = pk.load(handle)


def model(data, x_past, u_past, t_current):
    """
      Function that defines the Pyomo model instance for the supply chain network data.

      Inputs:
      - data: dict: Corresponding to all the supply chain network parameters (e.g., facilities, network connection,
      cost coefficients, lead times, etc.)
      - x_past: dict: Data corresponding to previous states (inventory and backorder) of the supply chain. That is, for
      time periods t less than t_current
      - u_past: dict: Data corresponding to previous control  inputs (shipments, production, sales, etc.) of the supply
      chain. This is used to account for shipments in the past that might be arriving in the future due to time delay
      - t_current: int: Corresponds to the present time at which the simulation is run

      Outputs:
      - m: pyo.ConcreteModel(): Pyomo instance of the supply chain management problem
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
        return data['demand'][i][t]

    m.d_it = pyo.Param(m.R, m.t_u, initialize=d_it_init, mutable=True)

    # Production capacity
    def _p_UB_init(m, i, t):
        return data['production_UB'][i][t]

    m.p_UB = pyo.Param(m.P, m.t_u, initialize=_p_UB_init)

    # Shipment costs
    def _s_c_init(m, i, j):
        return data['transportation_costs'][(i, j)]

    m.s_c = pyo.Param(m.A, initialize=_s_c_init)

    # Holding costs
    def _ih_c_init(m, i):
        return data['holding_costs'][i]

    m.ih_c = pyo.Param(m.N, initialize=_ih_c_init)

    # Production costs
    def _p_c_init(m, i):
        return data['production_costs'][i]

    m.p_c = pyo.Param(m.P, initialize=_p_c_init)

    # Backorder penalty
    # Backorder penalty
    def _bo_c_init(m, i):
        return data['backorder_penalty'][i]

    m.bo_c = pyo.Param(m.R, initialize=_bo_c_init)

    # Waste penalty
    def _w_c_init(m, i):
        return data['waste_penalty'][i]

    m.w_c = pyo.Param(m.N, initialize=_w_c_init)

    # Inventory with quality level q in facility i with temperature level k at beginning of period t
    m.I_iqkt = pyo.Var(m.N, m.q, m.k, m.t_x, within=pyo.NonNegativeReals)

    # Flow quantities on arcs (i,j) in period t, temperature level k with quality q starting from node i
    m.x_ijqkt = pyo.Var(m.A, m.q, m.k, m.t_u, within=pyo.NonNegativeReals, initialize=0)

    # Binary variable indicates whether the facility i has temperature level k in period t
    m.z_ikt = pyo.Var(m.N, m.k, m.t_x, within=pyo.Binary)

    # Number of batches with quality level q at the end of the period t required to be produced in plant i
    m.y_iqt = pyo.Var(m.P, m.q, m.t_u, within=pyo.NonNegativeReals)

    # Binary variable indicated whether transportation equipment on arc (i,j) has temperature level k in period t
    m.o_ijkt = pyo.Var(m.A, m.k, m.t_u, within=pyo.Binary)

    # Amount of waste in facility i in period t
    m.omega_it = pyo.Var(m.N, m.t_u, within=pyo.NonNegativeReals)

    # Amount sold in facility i at quality level 1 at in period t
    m.s_iqt = pyo.Var(m.R, m.q, m.t_u, within=pyo.NonNegativeReals)

    # Missed sales variable
    m.BO_it = pyo.Var(m.R, m.t_x, within=pyo.NonNegativeReals)

    # Fixing that nothing can be sold bellow the minimum quality required
    for i in m.R:
        for q in m.q:
            if q <= data['minimum_quality'][i]:
                for t in m.t_u:
                    m.s_iqt[i, q, t] = 0
                    m.s_iqt[i, q, t].fixed = True

    # Fixing that nothing arriving at a quality less than requires to zero
    # for i, j in m.A:
    #     for q in m.q:
    #         for k in m.k:
    #             for t in m.t_u:
    #                 if q < data['minimum_quality'][i] + data['degradation_ship'][(i, j)][k]:
    #                     m.x_ijqkt[i, j, q, k, t] = 0
    #                     m.x_ijqkt[i, j, q, k, t].fixed = True

    # Fixing that nothing can be produced with quality less than minimum quality at P
    for i in m.P:
        for q in m.q:
            # if q < data['minimum_quality'][i]:
            if q not in data['prod_distribution'].keys():
                for t in m.t_u:
                    m.y_iqt[i, q, t] = 0
                    m.y_iqt[i, q, t].fixed = True

    def _prod_quality_rule(m,i,q,t):
        if q in data['prod_distribution'].keys() and t == t_current:
            return data['prod_distribution'][q] * sum(m.y_iqt[i,q,t] for q in m.q) <= m.y_iqt[i,q,t]

        elif q in data['prod_distribution'].keys()  and t > t_current:
            return data['prod_distribution_nominal'][q] * sum(m.y_iqt[i, q, t] for q in m.q) <= m.y_iqt[i, q, t]
        else:
            return pyo.Constraint.Skip

    m.prod_quality = pyo.Constraint(m.P, m.q, m.t_u, rule = _prod_quality_rule)

    # Fixing that no inventory is stored under the minimum quality
    for i in m.N:
        for q in m.q:
            if q < data['minimum_quality'][i]:
                for k in m.k:
                    for t in m.t_x:
                        m.I_iqkt[i, q, k, t] = 0
                        m.I_iqkt[i, q, k, t].fixed = True

    # Cost minimization objective function
    # Production costs
    m.prod_costs = pyo.Var(m.t_u, within=pyo.NonNegativeReals)

    def _rule_prod_costs_c(m, t): # function for q dependent prod. costs: * (q / 100) ** 2
        return sum(sum(m.p_c[i] * m.y_iqt[i, q, t]   for q in m.q) for i in m.P) <= m.prod_costs[t]

    m.prod_costs_c = pyo.Constraint(m.t_u, rule=_rule_prod_costs_c)

    # Shipment costs
    m.ship_costs = pyo.Var(m.t_u, within=pyo.NonNegativeReals)

    def _rule_ship_costs_c(m, t):
        return sum(sum(sum(m.s_c[i, j] * (k**2) * m.x_ijqkt[i, j, q, k, t] for q in m.q) for i, j in m.A)
                   for k in m.k) <= m.ship_costs[t]

    m.ship_costs_c = pyo.Constraint(m.t_u, rule=_rule_ship_costs_c)

    # Inventory costs
    m.inv_costs = pyo.Var(m.t_x, within=pyo.NonNegativeReals)

    def _rule_inv_costs_c(m, t):
        return sum(sum(sum(m.ih_c[i] * (k**2) * m.I_iqkt[i, q, k, t] for q in m.q) for i in m.N)
                   for k in m.k) <= m.inv_costs[t]

    m.inv_costs_c = pyo.Constraint(m.t_x, rule=_rule_inv_costs_c)

    # Waste costs
    m.waste_costs = pyo.Var(m.t_u, within=pyo.NonNegativeReals)

    def _rule_waste_costs_c(m, t):
        return sum(m.w_c[i] * m.omega_it[i, t] for i in m.N) <= m.waste_costs[t]

    m.waste_costs_c = pyo.Constraint(m.t_u, rule=_rule_waste_costs_c)

    # Missed sales costs
    m.backorder_costs = pyo.Var(m.t_x, within=pyo.NonNegativeReals)


    def _rule_backorder_costs_c(m, t):
        return sum(m.bo_c[i] * m.BO_it[i, t] for i in m.R) <= m.backorder_costs[t]

    m.backorder_costs_c = pyo.Constraint(m.t_x, rule=_rule_backorder_costs_c)

    m.sales_profit = pyo.Var(m.t_u, within = pyo.NonNegativeReals)
    def _rule_sales_profit_c(m, t):
        return sum(sum(data['sales_profit'][(i,q)] * m.s_iqt[i,q,t] for i in m.R) for q in m.q) >= m.sales_profit[t]

    m.sales_profit_c = pyo.Constraint(m.t_u, rule = _rule_sales_profit_c)

    # Objective function: minimize total costs
    m.obj = pyo.Objective(expr=sum(m.prod_costs[t] + m.ship_costs[t] + m.waste_costs[t] for t in m.t_u) + \
                               sum(m.inv_costs[t] + m.backorder_costs[t] for t in m.t_x) + \
                               0 * sum(sum(sum(m.o_ijkt[i, j, k, t] for i, j in m.A) for k in m.k) for t in m.t_u) - \
                               (1/400)*sum(m.sales_profit[t] for t in m.t_u),
                          sense=pyo.minimize)

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
                             for k in m.k)
            else:
                return pyo.Constraint.Skip

    m.c9_1 = pyo.Constraint(m.P, m.q, m.t_u, rule=c9_rule1)

    # (m_true.I_iqkt['P1', 23, 1, 1].value + m_true.y_iqt['P1', 22, 1].value - m_true.x_ijqkt['P1', 'D1', 22, 1, 1].value)


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
                               q >= data['minimum_quality'][j] + data['degradation_ship'][(i, j)][k]) for k in m.k)
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
                               q + data['degradation_ship'][(j, i)][k] <= data['maximum_quality'][i]) for k in m.k) - \
                       m.s_iqt[i, q, t]
            else:
                return pyo.Constraint.Skip

    m.c10R_1 = pyo.Constraint(m.R, m.q, m.t_u, rule=c10R_rule1)

    # Backorder balance at retailers
    def c10R_rule2(m, i, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            if t + 1 <= pyo.value(m.H):
                return m.BO_it[i, t + 1] ==  m.d_it[i, t] - sum(
                    m.s_iqt[i, q, t] for q in m.q if q >= data['minimum_quality'][i]) + m.BO_it[i, t]
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


    # Assignment of transportation temperature
    def c13_rule1(m, i, j, k, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            return sum(m.x_ijqkt[i, j, q, k, t] for q in m.q if q >= data['minimum_quality'][i] + \
                   data['degradation_ship'][(i, j)][k]) <= m.M * m.o_ijkt[i, j, k, t]

    m.c13_1 = pyo.Constraint(m.A, m.k, m.t_u, rule=c13_rule1)

    def c14_rule1(m, i, j, k, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            return sum(m.x_ijqkt[i, j, q, k, t] for q in m.q if q >= data['minimum_quality'][i] + \
                   data['degradation_ship'][(i, j)][k]) >= m.o_ijkt[i, j, k, t]

    m.c14_1 = pyo.Constraint(m.A, m.k, m.t_u, rule=c14_rule1)

    def c15_rule(m, i, j, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            return sum(m.o_ijkt[i, j, k, t] for k in m.k) <= 1

    m.c15 = pyo.Constraint(m.A, m.t_u, rule=c15_rule)

    # Assignment of storage temperature
    def c16_rule1(m, i, q, k, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            return m.I_iqkt[i, q, k, t] <= m.M * m.z_ikt[i, k, t]

    m.c16_1 = pyo.Constraint(m.N, m.q, m.k, m.t_x, rule=c16_rule1)

    def c17_rule(m, i, t):
        if t < t_current:
            return pyo.Constraint.Skip
        else:
            return sum(m.z_ikt[i, k, t] for k in m.k) == 1

    m.c17 = pyo.Constraint(m.N, m.t_x, rule=c17_rule)

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

    # order_period = 5
    # def consecutive_ship_rule(m, i, j, t):
    #     if t < t_current:
    #         return pyo.Constraint.Skip
    #     if t+order_period >= data['H']:
    #         return pyo.Constraint.Skip
    #     else:
    #         return sum(sum(m.o_ijkt[i,j,k,t_in] for k in m.k) for t_in in range(t,t+order_period)) <= 1
    #
    # m.consecutive_ship = pyo.Constraint(m.A, m.t_u, rule = consecutive_ship_rule)

    # def c_bakcord_bound_rule(m,t):
    #     return m.BO_it['R1',t] <= 200
    #
    # m.c_bakcord_bound = pyo.Constraint( m.t_u, rule=c_bakcord_bound_rule )

    # Terminal constraint
    def c_term_rule1(m, i):
        if q == data['minimum_quality'][i] and i[0] == 'R':
            return sum(sum(m.I_iqkt[i, q, k, data['H']] for k in m.k) for q in m.q) == m.d_it[i, data['H'] - 1]
        elif q == data['minimum_quality'][i] and i in ['D1','P1'] :
            return sum(sum(m.I_iqkt[i, q, k, data['H']] for k in m.k) for q in m.q) == 4*m.d_it[i, data['H'] - 1]
        else:
            return sum(sum(m.I_iqkt[i, q, k, data['H']] for k in m.k) for q in m.q) <= 0.05

    m.c_term1 = pyo.Constraint(m.N, rule=c_term_rule1)

    def c_term_rule2(m, i):
        return m.BO_it[i, data['H']] == 0

    m.c_term2 = pyo.Constraint(m.R, rule=c_term_rule2)

    for i in x_past['inventory'].keys():
        m.I_iqkt[i[0], i[1], i[2], i[3]] = x_past['inventory'][i]
        m.I_iqkt[i[0], i[1], i[2], i[3]].fixed = True

    for i in x_past['backorder'].keys():
        m.BO_it[i[0], i[1]] = x_past['backorder'][i]
        m.BO_it[i[0], i[1]].fixed = True

    for i in u_past['shipment'].keys():
        m.x_ijqkt[i[0], i[1], i[2], i[3], i[4]] = u_past['shipment'][i]
        m.x_ijqkt[i[0], i[1], i[2], i[3], i[4]].fixed = True

    for i in u_past['production'].keys():
        m.y_iqt[i[0], i[1], i[2]] = u_past['production'][i]
        m.y_iqt[i[0], i[1], i[2]].fixed = True

    for i in u_past['sales'].keys():
        m.s_iqt[i[0], i[1], i[2]] = u_past['sales'][i]
        m.s_iqt[i[0], i[1], i[2]].fixed = True

    for i in u_past['waste'].keys():
        m.omega_it[i[0], i[1]] = u_past['waste'][i]
        m.omega_it[i[0], i[1]].fixed = True




        # # Demand satisfaction
    # def c22_rule(m, j, t):
    #     if t >= 3:
    #         return sum(m.s_iqt[j, q, t] for q in m.q if q >= data['minimum_quality'][i]) >= 0.3*m.d_it[i,t]
    #     else:
    #         return pyo.Constraint.Skip
    #
    # m.c22_1 = pyo.Constraint(m.R, m.t_u, rule=c22_rule)

    return m

