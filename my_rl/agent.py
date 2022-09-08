import numpy as np
import tensorflow as tf
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity, batch_size=64, for_conv=False):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.enough_data = False

    def add_data(self, data):
        self.memory.append(data)
        if len(self.memory) >= self.capacity:
            self.enough_data =True

    def sample_batch(self):
        batch_index = tf.random.uniform(shape=(self.batch_size, ), minval=0, maxval=self.capacity-1, dtype=tf.dtypes.int32)
        return np.array([self.memory[x] for x in batch_index])

    def clear_memory (self):
        self.memory.clear()
        self.enough_data = False

class Algorithm(object):
    def __init__(self, algorithm) -> None:
        self.algorithm = algorithm

    def learn(self, batches):
        return self.algorithm.learn(batches)

    def sample_action(self, state):
        return self.algorithm.sample_action(state)


class Agent(object):
    def __init__(self, replay_buffer:ReplayBuffer, algorithm:Algorithm, state_space, action_space, action_scale=1) -> None:
        self.replay_buffer = replay_buffer
        self.algorithm = algorithm
        self.state_space = state_space
        self.action_space = action_space
        self.action_scale = action_scale

    def train(self):
        batches = self.replay_buffer.sample_batch()
        return self.algorithm.learn(batches)

    def sample_action(self, state):
        action = self.algorithm.sample_action(state)*self.action_scale
        return tf.reshape(action, shape=(self.action_space,))