from collections import defaultdict
import numpy as np
import subprocess
import os
import sys
import networkx as nx
from copy import deepcopy
import re

sys.path.append(os.getcwd())
from src.misc.utils import mat2str

class MPC:
    def __init__(self, env, CPLEXPATH=None, platform = None, T = 20):
        self.env = env 
        self.T = T
        self.CPLEXPATH = CPLEXPATH
        if self.CPLEXPATH is None:
            self.CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        self.platform = platform
        
    def MPC_exact(self, modPath=None, MPCPath=None, forecast=False):
        t = self.env.time
        if forecast:
            demandAttr = [(i,j,tt,max([0, np.random.normal(self.env.scenario.demand_input[i,j][tt], 10) if tt > t else self.env.demand[i,j][tt]]),self.env.demandTime[i,j][tt], self.env.price[i,j][tt]) \
                          for i,j in self.env.demand for tt in range(t,t+self.T) if tt in self.env.demand[i,j] and self.env.scenario.demand_input[i,j][tt]>1e-3]
        else:
            demandAttr = [(i,j,tt,self.env.demand[i,j][tt],self.env.demandTime[i,j][tt], self.env.price[i,j][tt]) \
                          for i,j in self.env.demand for tt in range(t,t+self.T) if tt in self.env.demand[i,j] and self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,t+self.T)]
        edgeAttr = [(i,j,self.env.rebTime[i,j][t]) for i,j in self.env.edges]
        #rebEdgeAttr = [(i,j,self.env.rebTime[i,j][t]) for i,j in self.env.edges_reb]
        modPath = os.getcwd().replace('\\','/')+'/src/cplex_mod/' if modPath is None else modPath
        MPCPath = os.getcwd().replace('\\','/')+'/saved_files/cplex_logs/mpc/' if MPCPath is None else MPCPath
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + 'data_{}.dat'.format(t)
        resfile = MPCPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('t0='+str(t)+';\r\n')
            file.write('T='+str(self.T)+';\r\n')
            file.write('beta='+str(self.env.beta)+';\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            #file.write('rebEdgeAttr='+mat2str(rebEdgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('daccAttr='+mat2str(daccTuple)+';\r\n')
        
        modfile = modPath+'mpc.mod'
        my_env = os.environ.copy()
        if self.platform == None:
            my_env["LD_LIBRARY_PATH"] = self.CPLEXPATH
        else:
            my_env["DYLD_LIBRARY_PATH"] = self.CPLEXPATH
        out_file =  MPCPath + 'out_{}.dat'.format(t)
        with open(out_file,'w') as output_f:
            subprocess.check_call([self.CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        with open(resfile,'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)',')').strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,j,f1,f2 = v.split(',')
                        f1 = float(re.sub('[^0-9e.-]','', f1))
                        f2 = float(re.sub('[^0-9e.-]','', f2))
                        paxFlow[int(i),int(j)] = float(f1)
                        rebFlow[int(i),int(j)] = float(f2)
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.env.edges]
        rebAction = [rebFlow[i,j] if (i,j) in rebFlow else 0 for i,j in self.env.edges]
        return paxAction,rebAction
