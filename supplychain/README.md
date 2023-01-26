# Supply Chain Inventory Management

## Contents

* `src/algos/`: implementation of main algorithmic components
    * `/graph_rl_agent.py`: components for Graph-RL algorithms (PyTorch)
    * `/lcp_solver.py`: wrapper around the LCP formulation
    * `/mpc.py`: wrapper around the MPC used as "Oracle" benchmark
* `src/cplex_mod/`: formulations for optimization problems (CPLEX) 
    * `/lcp.mod`: LCP problem 
    * `/mpc.mod`: model predictive control formulation
* `src/envs/scim_env.py`: SCIM simulator 
* `src/misc/`: helper functions
* `saved_files/`: directory for saving results, logging, etc.

## Usage

Use `main_1f2s.py`, `main_1f3s.py`, and  `main_1f10s.py` to train and evaluate graph control algorithms on 1F2S, 1F3S, and 1F10S environments, respectively.

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
    --s_store       optimal store order-up-to-level 
    --s_factory     optimal factory order-up-to-level 
    
simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: 1010)
```

## Example

To train an agent (with the default parameters) run:

```bash
python main_1f3s.py
```

To train for a custom number of episodes, e.g., 55k:

```bash
python main_1f3s.py --max_episodes=55000
```

To evaluate a pretrained agent run the following:

```bash
python main_1f3s.py --test=True
```

To evaluate the domain-driven heuristic for 10 test episodes:

```bash
python main_1f3s.py --test=True --algo='heur' --max_episodes=10
```
