import numpy as np
import os, sys
import subprocess
from collections import defaultdict
import codecs

sys.path.append(os.getcwd())
from src.misc.utils import mat2str

def solveLCP(env, desiredDistrib=None, desiredProd=None, noiseTuple=None, sink=None, CPLEXPATH=None, res_path='scim', 
                 root_path='/mnt/array/daga_data/Github/SCIMAI-Gym'):
    t = env.time
    availableProd = [(i,max(env.acc[t-1][i] + env.arrival_prod[t][i],0)) for i in env.scenario.factory]
    desiredShip = [(env.scenario.warehouse[i],int(desiredDistrib[i]*sum([v for i,v in availableProd]))) for i in range(len(env.scenario.warehouse))]
    desiredProd = [(i, max(int(desiredProd[i].item()),0)) for i in env.scenario.factory]
    storageCapacity = [(i, env.scenario.storage_capacities[i]) for i in env.nodes]
    warehouseStock = [(i, env.acc[t-1][i] + env.arrival_flow[t][i]) for i in env.scenario.warehouse]
    edgeAttr = [(i,j,env.random_graph.edges[(i,j)]['time'], env.random_graph.edges[(i,j)]['cost']) for i,j in env.random_graph.edges]
    demand = [(i, env.demand[t][i]) for i in env.scenario.warehouse]
    modPath = os.getcwd().replace('\\','/')+'/src/cplex_mod/'
    matchingPath = os.getcwd().replace('\\','/')+'/saved_files/cplex_logs/' + res_path + '/'
    if not os.path.exists(matchingPath):
        os.makedirs(matchingPath)
    datafile = matchingPath + 'data_{}.dat'.format(t)
    resfile = matchingPath + 'res_{}.dat'.format(t)
    with open(datafile,'w') as file:
        file.write('path="'+resfile+'";\r\n')
        file.write('availableProd='+mat2str(availableProd)+';\r\n')
        file.write('desiredShip='+mat2str(desiredShip)+';\r\n')
        file.write('desiredProd='+mat2str(desiredProd)+';\r\n')
        file.write('storageCapacity='+mat2str(storageCapacity)+';\r\n')
        file.write('warehouseStock='+mat2str(warehouseStock)+';\r\n')
        file.write('demand='+mat2str(demand)+';\r\n')
        file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
    modfile = modPath+'lcp.mod'
    if CPLEXPATH is None:
        CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file =  matchingPath + 'out_{}.dat'.format(t)
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