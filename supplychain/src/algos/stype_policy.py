from collections import defaultdict

def s_type_policy(env, s_store, s_factory):
    t = env.time
    # compute the desired shipping quantities
    ship = dict()
    desiredStoreOrder = defaultdict()
    desiredStoreOrder_sum = 0
    # get desired shipping quantities for all stores 
    for (i,j) in env.G.edges:
        if j in env.scenario.warehouse:
            desiredStoreOrder[(i,j)] = s_store - (env.acc[t-1][j] + env.arrival_flow[t][j])
            desiredStoreOrder_sum += desiredStoreOrder[(i,j)]
    # if all store orders are feasible under the current factory availability: execute it
    if (env.acc[t-1][0] + env.arrival_prod[t-1][0]) >= desiredStoreOrder_sum:
        ship = desiredStoreOrder
    # otherwise, select store orders to maximize the minimum inventoy among all stores
    else:
        ratios = [desiredStoreOrder[0,j] / desiredStoreOrder_sum if j in env.scenario.warehouse else None for i,j in env.G.edges]
        for i, key in enumerate(desiredStoreOrder.keys()):
            ship[key] = (env.acc[t-1][key[0]] + env.arrival_prod[t][key[0]])*ratios[i]
    # compute the desired production quantities
    prod = dict()
    # compute available products at factory nodes
    av_factory = dict()
    for factory in env.scenario.factory:
        av_factory[factory] = env.acc[t-1][factory] + env.arrival_prod[t][factory] - sum([ship[key] for key in ship])
        diff = (s_factory - av_factory[factory])
        if max(0,diff) < env.scenario.storage_capacities[factory]:
            prod[factory] = max(0,diff)
        else:
            prod[factory] = env.scenario.storage_capacities[factory]
            
    return prod, ship
    