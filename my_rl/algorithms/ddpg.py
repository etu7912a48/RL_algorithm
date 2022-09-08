
import numpy as np
import tensorflow as tf
import copy

# net works
class CRITIC_NET(tf.keras.Model):
    init_ws = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    init_bs = tf.keras.initializers.RandomUniform(minval=3e-3, maxval=3e-3)

    def __init__(self):
        super().__init__()
        self.dense_i1 = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu)
        self.dense_i2 = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu)
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
        self.dense3 = tf.keras.layers.Dense(1, kernel_initializer=self.init_ws)

    def call(self, inputs):
        i1 = self.dense_i1(inputs[0])
        i2 = self.dense_i2(inputs[1])
        concat = self.concat([i1, i2])
        x = self.dense1(concat)
        y = self.dense2(x)
        z = self.dense3(y)
        return z

class ACTOR_NET(tf.keras.Model):
    init_ws = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    init_bs = tf.keras.initializers.RandomUniform(minval=3e-3, maxval=3e-3)

    def __init__(self, actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
        self.dense3 = tf.keras.layers.Dense(actions, activation=tf.keras.activations.tanh, kernel_initializer=self.init_ws)

    def call(self, inputs):
        x = self.dense1(inputs)
        x1 = self.dense2(x)
        return self.dense3(x1)

# stochastic method
class OUP(object):
    def __init__(self, action_shape, theta=0.15, sigma=0.2, dt=0.01):
        self.theta = theta
        self.mu = tf.zeros(shape=action_shape)
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.x = tf.zeros(shape=self.mu.shape)

    def __call__(self):
        self.x = self.x + (self.theta*(self.mu-self.x)*self.dt) + (self.sigma*tf.sqrt(self.dt)*tf.random.normal(shape=self.mu.shape))
        return self.x

# the core of ddpg
class DDPG(object):
    GAMMA = 0.99
    TAU = 0.001
    CRITIC_LEARNING_RATE = 1e-3
    ACTOR_LEARNING_RATE = 1e-4

    def __init__(self, action_dims, critic_net=None, actor_net=None) -> None:
        self.oup = OUP()
        self.critic_net = CRITIC_NET() if critic_net == None else critic_net
        self.critic_target_net = copy.deepcopy(self.critic_net)
        self.actor_net = ACTOR_NET() if actor_net == None else actor_net
        self.actor_target_net = copy.deepcopy(self.actor_net)
        self.critic_loss= tf.keras.losses.MeanSquaredError()
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.CRITIC_LEARNING_RATE)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.ACTOR_LEARNING_RATE)

    def sample_action(self, state):
        action = self.actor_net(state)+self.oup()
        return action

    def learn(self, batches):
        states = np.array([x[0] for x in batches], dtype=np.float32)
        actions = np.array([x[1] for x in batches], dtype=np.float32)
        rewards = np.array([x[2] for x in batches], dtype=np.float32)
        dones = np.array([x[3] for x in batches], dtype=np.float32)
        new_states = np.array([x[4] for x in batches], dtype=np.float32)

        self.calculate_critic_gradient(states, actions, rewards, dones, new_states)
        self.calculate_actor_gradient(states)

        self.update_nets(self.critic_target_net, self.critic_net)
        self.update_nets(self.actor_target_net, self.actor_net)


    # calculate the grads of critic net
    def calculate_critic_gradient(self, states, actions, rewards, dones, new_states):
        with tf.GradientTape() as tape:
            q_values = self.critic_net((states, actions), training=True)
            next_actions = self.actor_target_net(new_states, training=True)
            y = rewards + self.GAMMA*(1-dones)*self.critic_target_net((new_states, next_actions))
            c_l = tf.reduce_mean(self.critic_loss(q_values, y))
        c_grads = tape.gradient(c_l, self.critic_net.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic_net.trainable_weights))


    # calculate the grads  actor net
    def calculate_actor_gradient(self, states):
        with tf.GradientTape() as tape:
            current_actions = self.actor_net(states, training=True)
            current_q_values = self.critic_net((states, current_actions))
            a_l = -tf.reduce_mean(current_q_values)
        a_grads = tape.gradient(a_l, self.actor_net.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor_net.trainable_weights))

    def update_nets(self, target_net, net):
        for t_ws, ws in zip(target_net.weights, net.weights):
            t_ws.assign((1-self.TAU)*t_ws + self.TAU*ws)