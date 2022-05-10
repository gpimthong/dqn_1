from dqn import DQN
from buffer import ReplayBuffer

import numpy as np
import torch
import gym
from torch.utils.tensorboard import SummaryWriter
import datetime


Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)

env = gym.make('CartPole-v1')

config = {
    'dim_obs': 4,  # Q network input
    'dim_action': 2,  # Q network output
    'dims_hidden_neurons': (64, 64),  # Q network hidden
    'lr': 0.0005,  # learning rate
    'C': 60,  # copy steps
    'discount': 0.95,  # discount factor
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'eps_min': 0.01,
    'eps_max': 1.0,
    'eps_len': 4000,
    'seed': 1,
}

dqn = DQN(config)
buffer = ReplayBuffer(config)
train_writer = SummaryWriter()

steps = 0  # total number of steps
for i_episode in range(500):
    observation = env.reset()
    done = False
    t = 0  # time steps within each episode
    ret = 0.  # episodic return
    while done is False:
##        env.render()  # render to screen

        obs = torch.tensor(env.state)  # observe the environment state

        action = dqn.act_probabilistic(obs[None, :])  # take action

        next_obs, reward, done, info = env.step(action)  # environment advance to next step

        buffer.append_memory(obs=obs,  # put the transition to memory
                             action=torch.from_numpy(np.array([action])),
                             reward=torch.from_numpy(np.array([reward])),
                             next_obs=torch.from_numpy(next_obs),
                             done=done)

        dqn.update(buffer)  # agent learn

        t += 1
        steps += 1
        ret += reward  # update episodic return
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
        train_writer.add_scalar('Performance/episodic_return', ret, i_episode)  # plot

env.close()
train_writer.close()


