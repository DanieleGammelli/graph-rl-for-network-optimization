from __future__ import print_function
import argparse
import tqdm
from tqdm import trange
import numpy as np
import torch
import os
import sys

sys.path.append(os.getcwd())
from src.envs.dvr_env import Scenario, DVR
from src.algos.graph_rl_agent import A2C
from src.algos.lcp_solver import solveLCP
from src.misc.utils import dictsum
from src.algos.mpc import MPC

parser = argparse.ArgumentParser(description='A2C-GNN')

# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=int, default=9, metavar='S',
                    help='demand_ratio (default: 0.5)')
parser.add_argument('--json_hr', type=int, default=19, metavar='S',
                    help='json_hr (default: 7)')
parser.add_argument('--json_tsetp', type=int, default=4, metavar='S',
                    help='minutes per timestep (default: 3min)')
parser.add_argument('--beta', type=int, default=0.5, metavar='S',
                    help='cost of rebalancing (default: 0.5)')

# Model parameters
parser.add_argument('--algo', type=str, default='rl',
                    help='defines the algorithm to evaluate (only "rl" can use --test=False)')
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--cplexpath', type=str, default='/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/',
                    help='defines directory of the CPLEX installation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=30000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--max_steps', type=int, default=20, metavar='N',
                    help='number of steps per episode (default: T=60)')
parser.add_argument('--no-cuda', type=bool, default=True,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Define AMoD Simulator Environment
scenario = Scenario(json_file="data/scenario_nyc_brooklyn.json", sd=args.seed, demand_ratio=args.demand_ratio, json_hr=args.json_hr, json_tstep=args.json_tsetp, tf=20)
env = DVR(scenario, beta=args.beta)
if args.algo == 'rl':
    # Initialize A2C-GNN
    model = A2C(env=env, input_size=21).to(device)

if not args.test:
    #######################################
    #############Training Loop#############
    #######################################

    #Initialize lists for logging
    log = {'train_reward': [], 
           'train_served_demand': [], 
           'train_reb_cost': []}
    train_episodes = args.max_episodes #set max number of training episodes
    T = args.max_steps #set episode length
    epochs = trange(train_episodes) #epoch iterator
    best_reward = -np.inf #set best reward
    model.train() #set model in train mode

    for i_episode in epochs:
        obs = env.reset() #initialize environment
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        for step in range(T):
            # take matching step
            obs, paxreward, done, info = env.pax_step(CPLEXPATH=args.cplexpath, PATH='scenario_ny')
            episode_reward += paxreward
            # use Graph-RL policy (RL)
            action_rl = model.select_action(obs)
            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}
            # solve minimum rebalancing distance problem (LCP)
            rebAction = solveLCP(env,'scenario_ny',desiredAcc,args.cplexpath)
            # Take action in environment
            new_obs, rebreward, done, info = env.reb_step(rebAction)
            episode_reward += rebreward
            # Store the transition in memory
            model.rewards.append(paxreward + rebreward)
            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            # stop episode if terminating conditions are met
            if done:
                break
        # perform on-policy backprop
        model.training_step()

        # Send current statistics to screen
        epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}")
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(path=f"./saved_files/ckpt/ny/graph_rl.pth")
            best_reward = episode_reward
        # Log KPIs
        log['train_reward'].append(episode_reward)
        log['train_served_demand'].append(episode_served_demand)
        log['train_reb_cost'].append(episode_rebalancing_cost)
        model.log(log, path=f"./{args.directory}/rl_logs/ny/graph_rl.pth")
else:
    if args.algo == 'rl':
        # Load pre-trained model
        model.load_checkpoint(path=f"./{args.directory}/ckpt/ny/graph_rl.pth")
    if args.algo == 'oracle':
        mpc = MPC(env, CPLEXPATH=args.cplexpath, T=20)
    test_episodes = args.max_episodes #set max number of training episodes
    T = args.max_steps #set episode length
    epochs = trange(test_episodes) #epoch iterator
    #Initialize lists for logging
    log = {'test_reward': [], 
           'test_served_demand': [], 
           'test_reb_cost': []}
    task_reward_list = []
    for episode in epochs:
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        obs = env.reset()
        done = False
        k = 0
        while(not done):
            if args.algo == 'oracle':
                paxAction, rebAction = mpc.MPC_exact()
                obs, paxreward, done, info = env.pax_step(paxAction)
                obs, rebreward, done, info = env.reb_step(rebAction)
                episode_reward += paxreward + rebreward
            else:
            # take matching step (Step 1 in paper)
                obs, paxreward, done, info = env.pax_step(CPLEXPATH=args.cplexpath, PATH='scenario_ny_test')
                episode_reward += paxreward
                if args.algo == 'rl':
                    concentration, _ = model(obs)
                    action_rl = concentration / (concentration.sum() + 1e-16)
                if args.algo == 'heur':
                    action_rl = np.ones(env.nregion)/env.nregion
                if args.algo == 'random':
                    action_rl = np.random.dirichlet(np.ones(env.nregion))
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveLCP(env,'scenario_ny_test',desiredAcc,args.cplexpath)
                # Take action in environment
                new_obs, rebreward, done, info = env.reb_step(rebAction)
                episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            k += 1
        task_reward_list.append(episode_reward)
        # Send current statistics to screen
        epochs.set_description(f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost} | Aggregated: {np.mean(task_reward_list):.0f} +- {np.std(task_reward_list):.0f}")
    
    

