import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal, LogNormal, Poisson
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.utils import grid
from collections import namedtuple
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.99
args.log_interval = 10

#########################################
############## A2C PARSER ###############
#########################################

class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """
    def __init__(self, env, T=2, edge_list=None):
        super().__init__()
        self.env = env
        self.T = T
        self.edge_list = edge_list
        self.edge_list_bidirectional = edge_list + [(j,i) for (i,j) in edge_list]
        
    def parse_obs(self, obs):
        # Takes input from the environemnt and returns a graph representation of the data
        t = self.env.time
        # Node features
        x = torch.cat((torch.tensor([self.env.acc[t-1][n] for n in self.env.nodes]).view(1, 1, len(self.env.nodes)), # current available stock in all nodes
                       torch.tensor([obs[2][tt][n] for tt in range(t, t+3) for n in self.env.nodes]).view(1, 3, int(len(self.env.nodes))), # demand prediction for T=3 look-ahead
                       torch.tensor([self.env.arrival_flow[tt][n] for tt in range(t-1, t+self.T) for n in self.env.nodes]).view(1, 3, int(len(self.env.nodes))), # arriving flow in the next 3 timesteps
                       torch.tensor([self.env.arrival_prod[tt][n] for tt in range(t-1, t+self.T) for n in self.env.nodes]).view(1, 3, int(len(self.env.nodes))) # completed prod in the next 3 timesteps
        ), dim=1).view(10, len(self.env.nodes)).float().T
        
        # Edge features
        edge_index = torch.tensor(self.edge_list_bidirectional).T.long()
        e = torch.cat((
            torch.tensor([obs[0].edges[e]['time'] for e in self.edge_list*2]).view(1, edge_index.shape[1]).float(),
            torch.tensor([obs[0].edges[e]['cost'] for e in self.edge_list*2]).view(1, edge_index.shape[1]).float()
            ), dim=0).view(2, edge_index.shape[1]).T
        data = Data(x, edge_index, edge_attr=e)
        return data

class EdgeConv(MessagePassing):
    def __init__(self, node_size=4, edge_size=0, out_channels=4):
        super().__init__(aggr='add', flow="target_to_source") #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * node_size + edge_size, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

#########################################
############## A2C ACTOR ################
#########################################

class Actor(torch.nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """
    def __init__(self, node_size=4, edge_size=0, hidden_dim=32, out_channels=1, num_factories=1):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_factories = num_factories
        
        self.conv1 = EdgeConv(node_size, edge_size, hidden_dim)
        self.h_to_mu = nn.Linear(node_size + hidden_dim, out_channels)
        self.h_to_sigma = nn.Linear(node_size + hidden_dim, out_channels)
        self.h_to_concentration = nn.Linear(node_size + hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x_pp = self.conv1(x, edge_index, edge_attr)
        x_pp = torch.cat([x, x_pp], dim=1)
        
        mu, sigma = F.softplus(self.h_to_mu(x_pp[:self.num_factories])), F.softplus(self.h_to_sigma(x_pp[:self.num_factories]))
        alpha = F.softplus(self.h_to_concentration(x_pp[self.num_factories:]))
        return (mu, sigma), alpha

#########################################
############## A2C CRITIC ###############
#########################################

class Critic(torch.nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, node_size=4, edge_size=2, hidden_dim=32, out_channels=1):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.conv1 = EdgeConv(node_size, edge_size, hidden_dim)
        self.g_to_v = nn.Linear(node_size + hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x_pp = self.conv1(x, edge_index, edge_attr)

        x_pp = torch.cat([x, x_pp], dim=1)
        x_pp = torch.sum(x_pp, dim=0)

        v = self.g_to_v(x_pp)
        return v

#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, parser, actor, critic, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu"),
                clip=50, baseline=None, parametrization='Gaussian'):
        super(A2C, self).__init__()
        self.env = env
        self.parser = parser
        self.actor = actor
        self.critic = critic
        self.parametrization = parametrization
        
        self.optimizers = self.configure_optimizers()
        
        self.eps = eps
        self.device = device
        self.clip = clip if clip is not None else -1
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.baseline = baseline
        #self.env_baseline = env_baseline
        self.to(self.device)
        
    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        data = self.parser.parse_obs(obs)
        graph = data.to(self.device)
        
        # actor: computes concentration parameters of a X distribution
        a_probs = self.actor(graph.x, graph.edge_index, graph.edge_attr)

        # critic: estimates V(s_t)
        value = self.critic(graph.x, graph.edge_index, graph.edge_attr)
        return a_probs, value
    
    def select_action(self, obs, show_log_prob=False):
        if self.parametrization == 'Gaussian':
            (mu, sigma), value = self.forward(obs)
        
            m = Normal(loc=mu.view(-1,), scale=sigma.view(-1,))
        
        if self.parametrization == 'Poisson':
            rate, value = self.forward(obs)
        
            m = Poisson(rate=rate.view(-1,))
        
        if self.parametrization == 'Gaus-Dirichlet':
            a_probs , value = self.forward(obs)
            mu, sigma = a_probs[0][0], a_probs[0][1]
            alpha = a_probs[1] + 1e-16
            
            gaus = Normal(loc=mu.view(-1,), scale=sigma.view(-1,))
            dirichlet = Dirichlet(concentration=alpha.view(-1,))
        
        prod = gaus.sample()
        ship = dirichlet.sample()
        gaus_log_prob = gaus.log_prob(prod)
        dir_log_prob = dirichlet.log_prob(ship)
        self.saved_actions.append(SavedAction(gaus_log_prob + 0.05*dir_log_prob, value))
        if show_log_prob:
            return (prod, ship), (gaus_log_prob, dir_log_prob)
        return (prod, ship)

    def training_step(self, eps=1e-8):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).float()
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        if self.clip >= 0:
            a_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        if self.clip >= 0:
            v_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        if self.clip >= 0:
            return {"a_grad_norm": a_grad_norm, "v_grad_norm": a_grad_norm}
    
    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=5e-5)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=5e-5)
        return optimizers
    
    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)