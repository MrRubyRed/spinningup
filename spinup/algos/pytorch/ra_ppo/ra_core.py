import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount_minmax_overtime(l, g, gamma, v=None, debug=False):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vectors l, g
        [l0, [g0,
         l1,  g1,
         l2]  g2]

    output:
        [ gamma * max(g1, min(l1, gamma*max(g2, min(l2, gamma*max(g3,l3))))),
         (1-gamma)*max(l1,g1) + gamma * max(g1,min(l1, (1-gamma)*max(l2,g2) + )),
         gamma * max(g1, min(l1, gamma*max(l2,g2))),
         max(l2,g2)]
    """
    assert len(g) == len(l)
    v = l[-1] if v is None else v

    l_ = [(1.0 - gamma) * max(g[-1], l[-1])
          + gamma * max(g[-1], min(l[-1], v))]
    if len(l) == 1:
        return np.array(l_)
    assert ((len(l) - 2) >= 0)
    for ii in range(len(l)-2, -1, -1):
        l_.insert(0,
            (1.0 - gamma) * max(l[ii], g[ii]) + gamma * max(g[ii], min(l[ii], l_[0])))
    # Check that cost functional is correctly computed for gamma = 1
    if debug:
        g_ = np.copy(g)
        _l = np.copy(l)
        debug_list = []
        while len(g_) > 0:
            ep_ret = np.inf
            max_viol = -np.inf
            for ii in range(len(g_)):
                max_viol = max(max_viol, g_[ii])
                ep_ret = min(ep_ret, max(_l[ii], max_viol))
            debug_list.append(ep_ret)
            g_ = g_[1:]
            _l = _l[1:]
        print(l_)
        print(debug_list)
        plt.clf()
        plt.plot(l, 'g')
        # plt.plot(debug_list, 'b')
        plt.plot(g, 'k')
        plt.plot(l_, 'r')
        plt.pause(0.1)
        import pdb
        pdb.set_trace()

    return np.array(l_)


def discount_min_overtime(l, gamma):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vectors l, g
        [l0, [g0,
         l1,  g1,
         l2]  g2]

    output:
        [ gamma * max(g1, min(l1, gamma*max(g2, min(l2, gamma*max(g3,l3))))),
         (1-gamma)*max(l1,g1) + gamma * max(g1,min(l1, (1-gamma)*max(l2,g2) + )),
         gamma * max(g1, min(l1, gamma*max(l2,g2))),
         max(l2,g2)]
    """
    l_ = [l[-1]]
    if len(l) == 1:
        return np.array(l_)
    assert ((len(l) - 2) >= 0)
    for ii in range(len(l)-2, 0, -1):
        l_.insert(0, (1.0 - gamma) * l[ii] + gamma * min(l[ii], l_[0]))
    return np.array(l_)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, mix_term=0.05):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        d = min(act_dim*mix_term, 1.0)
        logits = d*(1.0/act_dim) + (1-d)*self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation,
                         output_activation=nn.Sigmoid)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            print("Creating CONTINUOUS Policy")
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            print("Creating DISCRETE Policy")
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
