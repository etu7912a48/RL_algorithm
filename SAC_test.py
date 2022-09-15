import gym
import numpy as np
import tensorflow as tf
from my_rl.agent import Agent, ReplayBuffer, Algorithm
from my_rl.algorithms.sac import SAC

game_name = 'BipedalWalker-v3'

# create env of the gam
env = gym.make(game_name, render_mode='human')
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
action_scale = (env.action_space.high[0] - env.action_space.low[0])/2

# create agent
rb = ReplayBuffer(capacity=10000)
alg = Algorithm(algorithm=SAC(state_space=state_space, action_space=action_space))
agent = Agent(replay_buffer=rb, algorithm=alg, action_scale=action_scale, state_space=state_space, action_space=action_space)

# start to train
agent.replay_buffer.clear_memory()
for episode in range(1000):
    observation, _ = env.reset(return_info=True)
    al_list = []
    reward_list = []

    for step in range(1000):
        action = agent.sample_action(observation.reshape(1, state_space))
        new_observation, reward, done, info = env.step(action)
        reward_list.append(reward)
        done = tf.cast(done, tf.float32)
        agent.replay_buffer.add_data(np.hstack((tf.squeeze(observation), tf.squeeze(action), reward, done, tf.squeeze(new_observation))))

        if agent.replay_buffer.enough_data:
            loss = agent.train()
            al_list.append(loss)

        observation = new_observation.copy()

        if done:
            break

    # report losses after every episode
    if episode%1 == 0 and agent.replay_buffer.enough_data:
        print(f'episode:{episode}, avg_al={tf.reduce_mean(al_list)} sum_r={tf.reduce_sum(reward_list)}')

env.close()