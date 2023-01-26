import numpy as np
import os, sys
import subprocess
from collections import defaultdict
import codecs

sys.path.append(os.getcwd())
from src.misc.utils import mat2str

def MPC(t, env, T = 10, dataPath=None, modPath=None, matchingPath=None, CPLEXPATH=None):
    datafile = os.getcwd().replace('\\','/')+'/saved_files/cplex_logs/mpc/' + f'data_{t}.dat'
    storageCost = [(n,env.random_graph.nodes[n]['storage_cost']) for n in env.random_graph]
    productionCost = [(n,env.random_graph.nodes[n]['production_cost']) for n in env.scenario.factory]
    storageCapacity = [(n,env.random_graph.nodes[n]['storage_capacity']) for n in env.random_graph]
    
    penalty_cost = env.scenario.penalty_cost
    prod_time = env.scenario.production_time
    prod_price = env.scenario.product_prices[0]
    availableProd = [(n, env.acc[t-1][n]) for n in env.scenario.factory ]
    warehouseStock = [(n, env.acc[t-1][n]) for n in env.scenario.warehouse ]
    demand = [(n,tt-t+1,env.demand[tt][n]) for tt in range(t,t+T) for n in env.scenario.warehouse]
    edgeAttr = [(i,j,env.random_graph.edges[i,j]['time'],env.random_graph.edges[i,j]['cost']) for i,j in env.random_graph.edges]
    arrivalTuple = [(n,tt-t+1,env.arrival_flow[tt][n]) for tt in range(t,t+T) for n in env.scenario.warehouse]
    expectedProdTuple = [(n,tt-t+1,env.arrival_prod[tt][n]) for tt in range(t,t+T) for n in env.scenario.factory]
    resfile = os.getcwd().replace('\\','/') + '/saved_files/cplex_logs/mpc/' + f'res_{t}.dat'
    modPath = os.getcwd().replace('\\','/')+ '/src/cplex_mod/' if modPath is None else modPath
    with open(datafile, 'w') as file:
        file.write('path="'+resfile+'";\r\n')
        file.write('T='+str(T)+';\r\n')
        file.write('penalty_cost='+str(penalty_cost)+';\r\n')
        file.write('prod_time='+str(prod_time)+';\r\n')
        file.write('product_price='+str(prod_price)+';\r\n')
        file.write('storageCost='+mat2str(storageCost)+';\r\n')
        file.write('storageCapacity='+mat2str(storageCapacity)+';\r\n')
        file.write('productionCost='+mat2str(productionCost)+';\r\n')
        file.write('availableProd='+mat2str(availableProd)+';\r\n')
        file.write('warehouseStock='+mat2str(warehouseStock)+';\r\n')
        file.write('demand='+mat2str(demand)+';\r\n')
        file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
        file.write('arrivalTuple='+mat2str(arrivalTuple)+';\r\n')
        file.write('expectedProdTuple='+mat2str(expectedProdTuple)+';\r\n')
    
    modfile = modPath+'mpc.mod'
    if CPLEXPATH is None:
        CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file =  os.getcwd().replace('\\','/')+'/saved_files/cplex_logs/mpc/' + 'out_{}.dat'.format(t)
    with open(out_file,'w') as output_f:
        subprocess.check_call([CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
    output_f.close()
    flow = defaultdict(float)
    production = defaultdict(float)
    with codecs.open(resfile,'r', encoding="utf8", errors="ignore") as file:
        for row in file:
            item = row.replace('e)',')').strip().strip(';').replace('?', '').replace('\x1f', '').replace('\x0f', '').replace('\x7f', '').replace('/', '').replace('O', '').split('=')
            if item[0] == 'flow':
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                        continue
                    i,j,f = v.split(',')
                    flow[int(i),int(j)] = float(f.replace('y6','').replace('I0','').replace('\x032','').replace('C8','').replace('C3','').replace('c5','').replace('#9','').replace('c9','').replace('\x132','').replace('c2','').replace('\x138','').replace('c2','').replace('\x133','').replace('\x131','').replace('s','').replace('#0','').replace('c4','').replace('\x031','').replace('c8','').replace('\x037','').replace('\x034','').replace('s4','').replace('S3','').replace('\x139','').replace('\x138','').replace('C4','').replace('\x039','').replace('S8','').replace('\x033','').replace('S5','').replace('#','').replace('\x131','').replace('\t6','').replace('\x01','').replace('i9','').replace('y4','').replace('a6','').replace('y5','').replace('\x018','').replace('I5','').replace('\x11','').replace('y2','').replace('\x011','').replace('y4','').replace('y5','').replace('a2','').replace('i9','').replace('i7','').replace('\t3','').replace('q','').replace('I3','').replace('A','').replace('y5','').replace('Q','').replace('a3','').replace('\x190','').replace('\x013','').replace('o', '').replace('`', '').replace('\x10', '').replace('P', '').replace('p', '').replace('@', '').replace('M', '').replace(']', '').replace('?', '').replace('\x1f', '').replace('}', '').replace('m', '').replace('\x04', '').replace('\x0f', '').replace('\x7f', '').replace('T', '').replace('$', '').replace('t', '').replace('\x147', '').replace('\x14', '').replace('\x046', '').replace('\x042', '').replace('/', '').replace('O', '').replace('D', '').replace('d', '').replace(')', '').replace('Y','').replace('i','').replace('\x193','').replace('\x192','').replace('y5','').replace('I2','').replace('\t','').replace('i2','').replace('!','').replace('i7','').replace('A8',''))
            if item[0] == 'production':
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                        continue
                    i,p = v.split(',')
                    production[int(i)] = float(p)
    ship = {(i,j): flow[i,j] if (i,j) in flow else 0 for i,j in env.G.edges}
    prod = {(i): production[i] if (i) in production else 0 for i in env.scenario.factory}
    action = (prod, ship)
    return action