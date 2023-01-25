# Dynamic Vehicle Routing
Official implementation of "Graph Reinforcement Learning for Network Control via Bi-Level Optimization"

## Contents

* `src/algos/`: implementation of main algorithmic components
    * `/graph_rl.py`: components for Graph-RL algorithms (PyTorch)
    * `/lcp_solver.py`: wrapper around the LCP formulation
    * `/mpc.py`: wrapper around the MPC used as "Oracle" benchmark
* `src/cplex_mod/`: formulations for optimization problems (CPLEX) 
    * `/lcp.mod`: LCP problem 
    * `/matching.mod`: passanger matching problem
    * `/mpc.mod`: model predictive control formulation
* `src/envs/dvr_env.py`: DVR simulator 
* `src/misc/`: helper functions
* `data/`: json files for NY and SHE experiments
* `saved_files/`: directory for saving results, logging, etc.

## Usage

Use `main_ny.py` and `main_she.py` to train and evaluate graph control algorithms on NY and SHE environments, respectively.

The code allows to test all algorithms evaluated in the paper via the following specification of the `algo` argument:

```bash
--algo='rl'         (Ours) Bi-level graph-RL agent
--algo='oracle'     Oracle MPC formulation
--algo='heur'       Domain-driven heuristic
--algo='random'     Randomized heuristic
```

Overall, the `main.py` files accept the following arguments:

```bash
cplex arguments:
    --cplexpath     defines directory of the CPLEX installation
    
model arguments:
    --algo          defines the algorithm to evaluate (only "rl" can be used with --test=False)
    --test          activates agent evaluation mode (default: False)
    --max_episodes  number of episodes to train agent (default: 30k)
    --max_steps     number of steps per episode (default: T=20)
    --no-cuda       disables CUDA training (default: True, i.e. run on CPU)
    --directory     defines directory where to log files (default: saved_files)
    
simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: NY=10, SHE=10)
    --demand_ratio  (default: NY=9, SHE=2.5)
    --json_hr       (default: NY=19, SHE=8)
    --json_tsetp    (default: NY=4, SHE=3)
    --no-beta       (default: NY=0.5, SHE=0.5)
```

## Example

To train an agent (with the default parameters) run:

```bash
python main_ny.py
```

To train for a custom number of episodes, e.g., 55k:

```bash
python main_ny.py --max_episodes=55000
```

To evaluate a pretrained agent run the following:

```bash
python main_ny.py --test=True
```

To evaluate the domain-driven heuristic for 10 test episodes:

```bash
python main_ny.py --test=True --algo='heur' --max_episodes=10
```
