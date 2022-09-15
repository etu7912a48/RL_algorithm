import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import copy

class CriticNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        concat = self.concat([inputs[0], inputs[1]])
        x = self.dense1(concat)
        x = self.dense2(x)
        return self.dense3(x)

class ActorNet(tf.keras.Model):
    def __init__(self, action_dims, clip_min=-1, clip_max=1, epsilon=1e-16):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.action_dims = action_dims
        self.epsilon = epsilon

        #mlp
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)

        # mean and std
        self.mean_dense = tf.keras.layers.Dense(self.action_dims)
        self.std_dense = tf.keras.layers.Dense(self.action_dims)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        y = self.mean_dense(x)
        z = self.std_dense(x)
        return y, z

    def eval(self, inputs):
        mean, log_std = self.call(inputs)
        std = tf.math.exp(log_std)
        n_dist = tfp.distributions.Normal(loc=mean, scale=std)

        z = mean + std*tf.random.normal(shape=(std.shape))
        actions = tf.tanh(z)
        log_prob = tf.reduce_sum(n_dist.log_prob(z) - tf.math.log(1 - tf.pow(actions, 2) + self.epsilon), 1, keepdims=True)
        return actions, log_prob

# the core of sac
class SAC(object):
    GAMMA = 0.99
    TAU = 0.003
    LEARNING_RATE = 3e-4

    def __init__(self, state_space, action_space, critic_net=None, actor_net=None) -> None:
        self.state_space = state_space
        self.action_space = action_space
        # net works
        self.actor_net = ActorNet(self.action_space) if actor_net == None else actor_net
        self.critic1_net = CriticNet() if critic_net == None else critic_net
        self.critic1_target_net = copy.deepcopy(self.critic1_net)
        self.critic2_net = CriticNet() if critic_net == None else critic_net
        self.critic2_target_net = copy.deepcopy(self.critic2_net)

        # loss and opt
        self.critic_loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)

        # temperature factor
        self.alpha = tf.Variable(0.0, dtype=tf.dtypes.float32)
        self.h0 = tf.constant(-action_space, dtype=tf.dtypes.float32)

    def sample_action(self, state):
        action, _ = self.actor_net.eval(state)
        return action

    def learn(self, batches):
        states = batches[:, 0:self.state_space]
        actions = batches[:, self.state_space:(self.state_space+self.action_space)]
        rewards = batches[:, (self.state_space+self.action_space):(self.state_space+self.action_space+1)]
        dones = batches[:, (self.state_space+self.action_space+1):(self.state_space+self.action_space+2)]
        new_states = batches[:, (self.state_space+self.action_space+2):]

        self.calculate_critic_gradient(states, actions, rewards, dones, new_states)
        a_l = self.calculate_actor_gradient(states)
        self.calculate_alpha_gradient(states)

        self.update_nets(self.critic1_target_net, self.critic1_net)
        self.update_nets(self.critic2_target_net, self.critic2_net)

        return a_l

    def calculate_critic_gradient(self, states, actions, rewards, dones, new_states):
        with tf.GradientTape() as tape:
            q1_values = self.critic1_net((states, actions))
            next_actions, log_prob = self.actor_net.eval(new_states)

            q1_target_values = self.critic1_target_net((new_states, next_actions))
            q2_target_values = self.critic2_target_net((new_states, next_actions))

            v = tf.minimum(q1_target_values, q2_target_values)-(self.alpha*log_prob)
            y = rewards + self.GAMMA*(1-dones)*v
            c1_l = tf.reduce_mean(0.5*self.critic_loss(q1_values, y))
        c1_grads = tape.gradient(c1_l, self.critic1_net.trainable_weights)

        with tf.GradientTape() as tape:
            q2_values = self.critic2_net((states, actions))
            next_actions, log_prob = self.actor_net.eval(new_states)

            q1_target_values = self.critic1_target_net((new_states, next_actions))
            q2_target_values = self.critic2_target_net((new_states, next_actions))

            v = tf.minimum(q1_target_values, q2_target_values)-(self.alpha*log_prob)
            y = rewards + self.GAMMA*(1-dones)*v
            c2_l = tf.reduce_mean(0.5*self.critic_loss(q2_values, y))
        c2_grads = tape.gradient(c2_l, self.critic2_net.trainable_weights)

        self.opt.apply_gradients(zip(c1_grads, self.critic1_net.trainable_weights))
        self.opt.apply_gradients(zip(c2_grads, self.critic2_net.trainable_weights))
        return (c1_l, c2_l)

    def calculate_actor_gradient(self, states):
        with tf.GradientTape() as tape:
            current_actions, log_prob = self.actor_net.eval(states)
            current_q1_values = self.critic1_net((states, current_actions))
            current_q2_values = self.critic2_net((states, current_actions))

            min_current_q_values = tf.minimum(current_q1_values, current_q2_values)
            a_l = tf.reduce_mean((self.alpha*log_prob)-min_current_q_values)
        a_grads = tape.gradient(a_l, self.actor_net.trainable_weights)
        self.opt.apply_gradients(zip(a_grads, self.actor_net.trainable_weights))
        return a_l

    def calculate_alpha_gradient(self, states):
        with tf.GradientTape() as tape:
            _, log_prob = self.actor_net.eval(states)
            alpha_l = -tf.reduce_mean(self.alpha*(log_prob+self.h0))
        alpha_grads = tape.gradient(alpha_l, [self.alpha])
        self.opt.apply_gradients(zip(alpha_grads, [self.alpha]))

    def update_nets(self, target_net, net):
        for t_ws, ws in zip(target_net.weights, net.weights):
            t_ws.assign((1-self.TAU)*t_ws + self.TAU*ws)