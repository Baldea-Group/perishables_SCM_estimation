def data_fun(H):
    """
    Function that generates supply chain instance data. This specific function was used to generate increasingly large
    supply chain network instances to validate the optimization problem

    Inputs:
    - H: int: corresponds to the horizon length of the optimization problem and is used for generating a demand vector of
     appropriate length

    Outputs:
    - data: dict: dictionary with all supply chain instance elements to define the optimization problem
    """
    data = {}

    network = 'central'

    # Optimization horizon
    data['H'] = H

    # Set of retailers
    R_num = 19
    data['R_set'] = ['R' + str(i) for i in range(1, R_num + 1)]

    # Set of distribution centers
    D_num = 6
    data['D_set'] = ['D'+str(i) for i in range(1,D_num+1)]

    # Set of producers
    P_num = 9
    data['P_set'] = ['P' + str(i) for i in range(1, P_num + 1)]

    # Generate routes in the supply chain
    routes = []
    for producer in data['P_set']:
        for distributor in data['D_set']:
            routes.append((producer, distributor))
    for distributor in data['D_set']:
        for retailer in data['R_set']:
            routes.append((distributor, retailer))

    # Update the data dictionary with the generated routes
    data['A_routes'] = routes

    # Predecessors to facility i
    n_i_init = {f: [] for f in data['P_set'] + data['D_set']}
    for p in data['P_set']:
        n_i_init[p] = data['D_set']
    for d in data['D_set']:
        n_i_init[d] = data['R_set']
    data['succ'] = n_i_init

    # Successors to facility i
    v_i_init = {f: [] for f in data['D_set'] + data['R_set']}
    for d in data['D_set']:
        v_i_init[d] = data['P_set']
    for r in data['R_set']:
        v_i_init[r] = data['D_set']
    data['pred'] = v_i_init

    # Supply chain shipment lead times
    data['ship_lead_time'] = {('P1', 'D1'): 5, ('P2', 'D1'): 2, ('P3', 'D1'): 5, ('P4', 'D1'): 3, ('P5', 'D1'): 4, ('P6', 'D1'): 2, ('P7', 'D1'): 4, ('P8', 'D1'): 3, ('P9', 'D1'): 5,
                              ('D1', 'R1'): 1, ('D1', 'R2'): 2, ('D1', 'R3'): 3, ('D1', 'R4'): 4, ('D1', 'R5'): 4, ('D1', 'R6'): 3, ('D1', 'R7'): 2, ('D1', 'R8'): 1, ('D1', 'R9'): 3, ('D1', 'R10'): 4,
                              ('D1', 'R11'): 3, ('D1', 'R12'): 2, ('D1', 'R13'): 1, ('D1', 'R14'): 3, ('D1', 'R15'): 2, ('D1', 'R16'): 4, ('D1', 'R17'): 1, ('D1', 'R18'): 2, ('D1', 'R19'): 3,
                              ('P1', 'D2'): 1, ('P2', 'D2'): 2, ('P3', 'D2'): 1, ('P4', 'D2'): 3, ('P5', 'D2'): 4, ('P6', 'D2'): 2, ('P7', 'D2'): 3, ('P8', 'D2'): 4, ('P9', 'D2'): 1,
                              ('D2', 'R1'): 1, ('D2', 'R2'): 2, ('D2', 'R3'): 3, ('D2', 'R4'): 4, ('D2', 'R5'): 2, ('D2', 'R6'): 1, ('D2', 'R7'): 3, ('D2', 'R8'): 2, ('D2', 'R9'): 1, ('D2', 'R10'): 4,
                              ('D2', 'R11'): 3, ('D2', 'R12'): 2, ('D2', 'R13'): 1, ('D2', 'R14'): 4, ('D2', 'R15'): 3, ('D2', 'R16'): 1, ('D2', 'R17'): 2, ('D2', 'R18'): 3, ('D2', 'R19'): 4,
                              ('P1', 'D3'): 3, ('P2', 'D3'): 4, ('P3', 'D3'): 3, ('P4', 'D3'): 2, ('P5', 'D3'): 1, ('P6', 'D3'): 4, ('P7', 'D3'): 2, ('P8', 'D3'): 1, ('P9', 'D3'): 3,
                              ('D3', 'R1'): 2, ('D3', 'R2'): 3, ('D3', 'R3'): 2, ('D3', 'R4'): 1, ('D3', 'R5'): 3, ('D3', 'R6'): 4, ('D3', 'R7'): 1, ('D3', 'R8'): 3, ('D3', 'R9'): 4, ('D3', 'R10'): 2,
                              ('D3', 'R11'): 1, ('D3', 'R12'): 3, ('D3', 'R13'): 4, ('D3', 'R14'): 2, ('D3', 'R15'): 1, ('D3', 'R16'): 3, ('D3', 'R17'): 4, ('D3', 'R18'): 2, ('D3', 'R19'): 1,
                              ('P1', 'D4'): 4, ('P2', 'D4'): 3, ('P3', 'D4'): 2, ('P4', 'D4'): 1, ('P5', 'D4'): 3, ('P6', 'D4'): 2, ('P7', 'D4'): 1, ('P8', 'D4'): 3, ('P9', 'D4'): 4,
                              ('D4', 'R1'): 2, ('D4', 'R2'): 3, ('D4', 'R3'): 4, ('D4', 'R4'): 1, ('D4', 'R5'): 2, ('D4', 'R6'): 3, ('D4', 'R7'): 4, ('D4', 'R8'): 1, ('D4', 'R9'): 2, ('D4', 'R10'): 3,
                              ('D4', 'R11'): 4, ('D4', 'R12'): 1, ('D4', 'R13'): 2, ('D4', 'R14'): 3, ('D4', 'R15'): 4, ('D4', 'R16'): 1, ('D4', 'R17'): 2, ('D4', 'R18'): 3, ('D4', 'R19'): 4,
                              ('P1', 'D5'): 2, ('P2', 'D5'): 3, ('P3', 'D5'): 4, ('P4', 'D5'): 1, ('P5', 'D5'): 2, ('P6', 'D5'): 3, ('P7', 'D5'): 4, ('P8', 'D5'): 1, ('P9', 'D5'): 2,
                              ('D5', 'R1'): 3, ('D5', 'R2'): 4, ('D5', 'R3'): 1, ('D5', 'R4'): 2, ('D5', 'R5'): 3, ('D5', 'R6'): 4, ('D5', 'R7'): 1, ('D5', 'R8'): 2, ('D5', 'R9'): 3, ('D5', 'R10'): 4,
                              ('D5', 'R11'): 1, ('D5', 'R12'): 2, ('D5', 'R13'): 3, ('D5', 'R14'): 4, ('D5', 'R15'): 1, ('D5', 'R16'): 2, ('D5', 'R17'): 3, ('D5', 'R18'): 4, ('D5', 'R19'): 1,
                              ('P1', 'D6'): 4, ('P2', 'D6'): 1, ('P3', 'D6'): 3, ('P4', 'D6'): 2, ('P5', 'D6'): 4, ('P6', 'D6'): 1, ('P7', 'D6'): 3, ('P8', 'D6'): 2, ('P9', 'D6'): 1,
                              ('D6', 'R1'): 4, ('D6', 'R2'): 3, ('D6', 'R3'): 2, ('D6', 'R4'): 1, ('D6', 'R5'): 4, ('D6', 'R6'): 3, ('D6', 'R7'): 2, ('D6', 'R8'): 1, ('D6', 'R9'): 4, ('D6', 'R10'): 3,
                              ('D6', 'R11'): 2, ('D6', 'R12'): 1, ('D6', 'R13'): 4, ('D6', 'R14'): 3, ('D6', 'R15'): 2, ('D6', 'R16'): 1, ('D6', 'R17'): 4, ('D6', 'R18'): 3, ('D6', 'R19'): 2}


    # Production lead time
    data['prod_lead_time'] = {i: 0 for i in data['P_set']}

    # Consumer demand at retail facility
    data['demand'] = {i: [40 for t in range(data['H'] + 1)] for i in data['R_set']}

    # Holding costs
    data['holding_costs'] = {}
    for i in data['P_set']:
        data['holding_costs'][i] = 3
    for i in data['D_set']:
        data['holding_costs'][i] = 0.5
    for i in data['R_set']:
        data['holding_costs'][i] = 2


    # Transportation costs (per unit of inventory per day for a given supply chain route)
    data['transportation_costs'] = {(i, j): 2 for i, j in data['A_routes']}
    # Production costs
    data['production_costs'] = {'P1': 0.1, 'P2': 4, 'P3': 2, 'P4': 3, 'P5': 2.5, 'P6': 3.5, 'P7': 0.1, 'P8': 4, 'P9': 2}
    # Backorder penalty
    data['backorder_penalty'] = {i: 50 for i in data['R_set']}
    data['backorder_penalty']['R2'] = 54
    data['backorder_penalty']['R3'] = 58
    data['backorder_penalty']['R4'] = 82
    data['backorder_penalty']['R5'] = 52
    data['backorder_penalty']['R6'] = 90
    data['backorder_penalty']['R7'] = 55
    data['backorder_penalty']['R8'] = 59
    data['backorder_penalty']['R9'] = 83
    data['backorder_penalty']['R10'] = 56
    data['backorder_penalty']['R11'] = 60
    data['backorder_penalty']['R12'] = 84
    data['backorder_penalty']['R13'] = 53
    data['backorder_penalty']['R14'] = 70
    data['backorder_penalty']['R15'] = 65
    data['backorder_penalty']['R16'] = 59
    data['backorder_penalty']['R17'] = 53
    data['backorder_penalty']['R18'] = 78
    data['backorder_penalty']['R19'] = 51
    # Waste costs
    data['waste_penalty'] = {i: 25 for i in data['R_set'] + data['D_set'] + data['P_set']}

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
    for i, j in data['A_routes']:
        data['degradation_ship'][(i, j)] = {}
        for k in range(1, data['temperature'] + 1):
            data['degradation_ship'][(i, j)][k] = data['degradation'][k] * data['ship_lead_time'][(i, j)]

    # Minimum and maximum quality requirement for all facilities
    data['minimum_quality'] = {i: 1 for i in data['P_set'] + data['D_set'] + data['R_set']}
    data['maximum_quality'] = {i: data['quality'] for i in data['P_set'] + data['D_set'] + data['R_set']}

    # Sales costs as a function of quality
    data['sales_profit'] = {}
    for q in range(1, data['quality'] + 1):
        data['sales_profit'][('R1', q)] = q ** 0.5
        data['sales_profit'][('R2', q)] = q
        data['sales_profit'][('R3', q)] = q ** 3
        data['sales_profit'][('R4', q)] = q ** 2
        data['sales_profit'][('R5', q)] = q ** 1.5
        data['sales_profit'][('R6', q)] = q
        data['sales_profit'][('R7', q)] = q ** 2.5
        data['sales_profit'][('R8', q)] = q ** 1.8
        data['sales_profit'][('R9', q)] = q ** 2.2
        data['sales_profit'][('R10', q)] = q ** 2.7
        data['sales_profit'][('R11', q)] = q ** 1.9
        data['sales_profit'][('R12', q)] = q ** 2.3
        data['sales_profit'][('R13', q)] = q ** 2.6
        data['sales_profit'][('R14', q)] = q ** 0.5
        data['sales_profit'][('R15', q)] = q
        data['sales_profit'][('R16', q)] = q ** 3
        data['sales_profit'][('R17', q)] = q ** 2
        data['sales_profit'][('R18', q)] = q ** 1.5
        data['sales_profit'][('R19', q)] = q

    return data
