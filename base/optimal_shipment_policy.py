import pyomo.environ as pyo
import numpy as np


def policy(data, x_past_true_inventory, u_past_shipment):
    """
    Function that defines the Pyomo model instance for the optimal production and distribution policy computation.

      Inputs:
      - data: dict: Corresponding to all the supply chain network parameters (e.g., facilities, network connection,
      cost coefficients, lead times, etc.)
      - x_past_true_inventory: dict: Data corresponding to previous true states (inventory and backorder) of the supply
      chain. That is, fortime periods t less than t_current
      - u_past_shipment: dict: Dictionary of past shipments in the supply chain

      Outputs:
      - m: pyo.ConcreteModel(): Pyomo instance of the optimal production and distribution policy
    """

    m = pyo.ConcreteModel()
    # Quality index
    m.q = pyo.RangeSet(1, data['quality'])
    # Temperature index
    m.k = pyo.RangeSet(1, data['temperature'])

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

    # Flow quantities on arcs (i,j), temperature level k with quality q starting from node i
    m.x_ijqk = pyo.Var(m.A, m.q, m.k, within=pyo.NonNegativeReals, initialize=0)
    m.aux_var1 = pyo.Var(m.A, within=pyo.NonNegativeReals, initialize = 0 )

    penalty_q = {}
    for q1 in m.q:
        for q2 in m.q:
            if q1 == q2:
                penalty_q[(q1,q2)] = 1
            else:
                penalty_q[(q1, q2)] = (q1-q2)**4 + 1

    # Objective function
    def _obj_rule(m):
        return sum(sum(sum(sum(penalty_q[(q1, q2)] * (u_past_shipment[(i, j, q1, k)] - m.x_ijqk[i, j, q2, k])
                               for q2 in m.q) for q1 in m.q) for k in m.k) for i, j in m.A) + 1000000*sum(m.aux_var1[i,j] for i,j in m.A)**2

    m.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    def _inv_balance_rule(m, i, q):
        if i[0] == 'R':
            return pyo.Constraint.Skip
        else:
            return sum(x_past_true_inventory[(i, q + data['degradation'][k], k)] for k in m.k if
                       q + data['degradation'][k] <= data['maximum_quality'][i]) >= sum(
                sum(m.x_ijqk[i, j, q, k] for k in m.k) for j in m.n_i[i])

    m.inv_balance = pyo.Constraint(m.N, m.q, rule=_inv_balance_rule)

    def _net_flow_rule(m, i, j):
        return sum(sum(m.x_ijqk[i, j, q, k] for k in m.k) for q in m.q)  == sum(sum(u_past_shipment[(i, j, q, k)]  for k in m.k) for q in m.q) -  m.aux_var1[i,j]

    m.net_flow = pyo.Constraint(m.A, rule = _net_flow_rule)


    return m

