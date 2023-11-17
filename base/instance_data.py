def data_fun(H: int):
    """
    Function that generates supply chain instance data

    Inputs:
    - H: int: corresponds to the horizon length of the optimization problem and is used for generating a demand vector of appropriate length

    Outputs:
    - data: dict: dictionary with all supply chain instance elements to define the optimization problem
    """
    data = {}

    network = 'central'

    # Optimization horizon
    data['H'] = H

    # Set of retailers
    data['R_set'] = ['R1', 'R2', 'R3', 'R4']

    # Set of distribution centers
    data['D_set'] = ['D1']

    # Set of producers
    data['P_set'] = ['P1', 'P2']

    # Routes in the supply chain
    data['A_routes'] = [('P1', 'D1'), ('P2', 'D1'), ('D1', 'R1'), ('D1', 'R2'), ('D1', 'R3'), ('D1', 'R4')]

    # Predecessors to facility i
    n_i_init = {}
    n_i_init['P1'] = ['D1']
    n_i_init['P2'] = ['D1']
    n_i_init['D1'] = ['R' + str(i) for i in range(1, 5)]
    data['succ'] = n_i_init

    # Successors to facility i
    v_i_init = {}
    v_i_init['D1'] = ['P1', 'P2']
    for i in ['R' + str(i) for i in range(1, 5)]:
        v_i_init[i] = ['D1']
    data['pred'] = v_i_init

    # Supply chain shipment lead times
    data['ship_lead_time'] = {('P1', 'D1'): 5, ('P2', 'D1'): 2, ('D1', 'R1'): 1, ('D1', 'R2'): 2, ('D1', 'R3'): 3,
                              ('D1', 'R4'): 4}
    # Production lead time
    data['prod_lead_time'] = {'P1': 0, 'P2': 0}

    # Consumer demand at retail facility
    data['demand'] = {i: [40 for t in range(data['H']+1)] for i in data['R_set']}




    # Holding costs
    data['holding_costs'] = {'P1': 3, 'P2': 3, 'D1': .5, 'R1': 2, 'R2': 2, 'R3': 2, 'R4': 2}
    # Transportation costs (per unit of inventory per day for a given supply chain route)
    data['transportation_costs'] = {(i, j): 2 for i, j in data['A_routes']}
    # Production costs
    data['production_costs'] = {'P1': 0.1, 'P2': 4}
    # Backorder penalty
    data['backorder_penalty'] = {i:50 for i in data['R_set']}
    data['backorder_penalty']['R2'] = 54
    data['backorder_penalty']['R3'] = 58
    data['backorder_penalty']['R4'] = 82
    # Waste costs
    data['waste_penalty'] = {i:25 for i in data['R_set'] + data['D_set'] + data['P_set']}



    # DATA FOR MODEL CONSTRAINTS
    # Inventory upper bound
    data['inventory_UB'] = {i: [50000 for _ in range(data['H'])] for i in data['R_set'] + data['D_set'] + data['P_set']}
    # Shipments upper bound
    data['shipment_UB'] = {(i, j): [500000 for _ in range(data['H'])] for i, j in data['A_routes']}
    # Production upper bound (production capacity)
    data['production_UB'] = {i: [400000 for _ in range(data['H'])] for i in data['P_set']}

    if network == 'central':
        pass
    else:
        data['R_set'] = [i for i in data['R_set'] if i in network]
        data['D_set'] = [i for i in data['D_set'] if i in network]
        data['P_set'] = [i for i in data['P_set'] if i in network]
        data['A_routes'] = [(i, j) for i, j in data['A_routes'] if i in network and j in network]
        data['succ'] = {i: [k for k in j if k in network] for i, j in data['succ'].items() if i in network}
        data['pred'] = {i: [k for k in j if k in network] for i, j in data['pred'].items() if i in network}
        data['ship_lead_time'] = {i: j for i, j in data['ship_lead_time'].items() if i in data['A_routes']}
        data['prod_lead_time'] = {i: j for i, j in data['prod_lead_time'].items() if i in data['P_set']}
        data['demand'] = {i: j for i, j in data['demand'].items() if i[0] in data['R_set']}
        data['holding_costs'] = {i: j for i, j in data['holding_costs'].items() if i in network}
        data['transportation_costs'] = {i: j for i, j in data['transportation_costs'].items() if i in data['A_routes']}
        data['production_costs'] = {i: j for i, j in data['production_costs'].items() if i in data['P_set']}
        data['inventory_UB'] = {i: j for i, j in data['inventory_UB'].items() if i in network}
        data['shipment_UB'] = {i: j for i, j in data['shipment_UB'].items() if i in data['A_routes']}
        data['production_UB'] = {i: j for i, j in data['production_UB'].items() if i in data['P_set']}


    # PERISHABILITY DATA
    # Number of allowed temperature levels (1 coldest 4 warmest)
    data['temperature'] = 1
    # Number of quality levels in the model
    data['quality'] = 30
    # Degradation rate as a function of temperature level (for single time period in storage)
    data['degradation'] = {1: 1, 2: 2, 3: 3, 4: 5}

    # Degradation units during the duration of shipment
    data['degradation_ship'] = {}
    for i,j in data['A_routes']:
        data['degradation_ship'][(i,j)] ={}
        for k in range(1, data['temperature']+1):
            data['degradation_ship'][(i, j)][k] = data['degradation'][k]*data['ship_lead_time'][(i,j)]

    # Minimum and maximum quality requirement for all facilities
    data['minimum_quality'] = {i:1 for i in data['P_set']+data['D_set']+data['R_set']}
    data['maximum_quality'] = {i:data['quality'] for i in data['P_set']+data['D_set']+data['R_set']}

    # Sales costs as a function of quality
    data['sales_profit'] = {}
    for q in range(1, data['quality']+1):
        data['sales_profit'][('R1', q)] = q ** 0.5
        data['sales_profit'][('R2', q)] = q
        data['sales_profit'][('R3', q)] = q ** 3
        data['sales_profit'][('R4', q)] = q ** 2

    return data






