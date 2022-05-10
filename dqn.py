import torch
import torch.nn as nn

from typing import Tuple
from numpy.random import binomial
from numpy.random import choice

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)


class DQN:
    def __init__(self, config):

        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.C = config['C']  # copy steps
        self.eps_len = config['eps_len']  # length of epsilon greedy exploration
        self.eps_max = config['eps_max']
        self.eps_min = config['eps_min']
        self.discount = config['discount']  # discount factor
        self.batch_size = config['batch_size']  # mini batch size

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.Q = QNetwork(dim_obs=self.dim_obs,
                          dim_action=self.dim_action,
                          dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q_tar = QNetwork(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons)

        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.training_step = 0

    def update(self, buffer):
        t = buffer.sample(self.batch_size)

        s = t.obs
        a = t.action
        r = t.reward
        sp = t.next_obs
        done = t.done

        tau = 1e-3

        self.training_step += 1

        Q_targets_next = self.Q_tar(sp.to(torch.float64)).detach().max(1)[0].unsqueeze(1)
        Q_targets = r + (self.discount * Q_targets_next * (~ done))
        Q_expected = self.Q(s.to(torch.float64)).gather(1, a.to(torch.int64))
        loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer_Q.zero_grad()
        loss.backward()
        self.optimizer_Q.step()

        for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def act_probabilistic(self, observation: torch.Tensor):
        # epsilon greedy:
        first_term = self.eps_max * (self.eps_len - self.training_step) / self.eps_len
        eps = max(first_term, self.eps_min)

        explore = binomial(1, eps)

        if explore == 1:
            a = choice(self.dim_action)
        else:
            self.Q.eval()
            Q = self.Q(observation)
            val, a = torch.max(Q, axis=1)
            a = a.item()
            self.Q.train()
        return a

    def act_deterministic(self, observation: torch.Tensor):
        self.Q.eval()
        Q = self.Q(observation)
        val, a = torch.max(Q, axis=1)
        self.Q.train()
        return a.item()


class QNetwork(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        if not isinstance(dim_obs, int):
            TypeError('dimension of observation must be int')
        if not isinstance(dim_action, int):
            TypeError('dimension of action must be int')
        if not isinstance(dims_hidden_neurons, tuple):
            TypeError('dimensions of hidden neurons must be tuple of int')

        super(QNetwork, self).__init__()
        self.num_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        n_neurons = (dim_obs, ) + dims_hidden_neurons + (dim_action, )
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, observation: torch.Tensor):
        x = observation
        for i in range(self.num_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        return self.output(x)

