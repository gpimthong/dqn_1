import gym
import tensorflow as tf

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import datetime


env = gym.make('CartPole-v1')
policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[64, 64])
model = DQN(MlpPolicy,
            policy_kwargs=policy_kwargs,
            env=env,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=10000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            exploration_initial_eps=1.0,
            batch_size=32,
            double_q=False,
            learning_starts=1,
            target_network_update_freq=30,
            tensorboard_log='tensorboard/baseline_dqn_{date:%Y-%m-%d_%H:%M:%S}'.format(
                             date=datetime.datetime.now()))
model.learn(total_timesteps=100000)

# model save & load
# model.save("deepq_cartpole")
# del model  # remove to demonstrate saving and loading
# model = DQN.load("deepq_cartpole")

# test model
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


