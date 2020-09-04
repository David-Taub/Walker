import logging
import tensorflow as tf
import keyboard
import numpy as np
# import matplotlib.pyplot as plt

from tensorflow.keras import layers
# from tensorflow.keras import activations

UPPER_BOUND = 1
LOWER_BOUND = -1
ACTION_REGULARIZATION = 0
ACTION_L2_REG_FACTOR = 0
CRITIC_L2_REG_FACTOR = 0.1


class OUActionNoise:
    """
    Ornstein-Uhlenbeck process
    """

    def __init__(self, output_size, mean=0, std_deviation=0.3, theta=5, dt=0.01):
        self.theta = theta
        self.mean = mean * np.ones(output_size)
        self.previous_x = mean
        self.std_dev = std_deviation
        self.dt = dt

    def __call__(self):

        # x = (self.previous_x + self.theta * (self.mean - self.previous_x) * self.dt + (
        #     self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)))
        x = self.previous_x + self.dt * (
            self.theta * (self.mean - self.previous_x) + self.std_dev * np.random.normal(size=self.mean.shape))
        self.previous_x = x
        return x


class MarkovSaltPepperNoise:
    def __init__(self, output_size, salt_to_pepper=0.02, pepper_to_salt=0.1):
        self.salt_to_pepper = salt_to_pepper
        self.pepper_to_salt = pepper_to_salt
        self.noise = np.ones(output_size)

    # def _reward_to_probabilty(self, reward):
    #     MAX_PROB = 0.05
    #     MIN_PROB = 0.001
    #     MIN_REWARD = 1
    #     MAX_REWARD = 7
    #     reward = min(MAX_REWARD, max(reward, MIN_REWARD))
    #     reward_streched = (reward - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)
    #     return MIN_PROB + (1 - reward_streched) * (MAX_PROB - MIN_PROB)

    def __call__(self):
        # self.salt_to_pepper = self._reward_to_probabilty(reward)
        switch_salt_to_pepper = np.random.binomial(1, self.salt_to_pepper, self.noise.shape)
        switch_pepper_to_salt = np.random.binomial(1, self.pepper_to_salt, self.noise.shape)
        self.noise[(self.noise == 1) & (switch_salt_to_pepper == 1)] = np.random.uniform(low=-1.5, high=-0.5)
        self.noise[(self.noise != 1) & (switch_pepper_to_salt == 1)] = 1
        return self.noise


class Buffer:
    def __init__(self, state_size, action_size, gamma, buffer_capacity=100000, batch_size=64):

        # self.im = plt.imshow(np.zeros((77, 36)), cmap='gray', vmin=-0.5, vmax=0.5)
        self.gamma = gamma
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_size))

    def record(self, observation):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = observation[0]
        self.action_buffer[index] = observation[1]
        self.reward_buffer[index] = observation[2]
        self.next_state_buffer[index] = observation[3]

        self.buffer_counter += 1

    def calc_critic_loss(self, indices, target_actor, critic_model, target_critic):
        state_batch = self.state_buffer[indices]
        action_batch = self.action_buffer[indices]
        reward_batch = self.reward_buffer[indices]
        next_state_batch = self.next_state_buffer[indices]
        target_actions = target_actor(next_state_batch)
        y = reward_batch + self.gamma * target_critic([next_state_batch, target_actions])
        critic_value = critic_model([state_batch, action_batch])
        critic_losses = tf.math.square(y - critic_value) + CRITIC_L2_REG_FACTOR * tf.math.square(critic_value)
        return critic_losses

    def calc_sample_probabilities(self, losses):
        """
        Rank variant of "Prioritized Experience Replay", Schaul et al. 2015 of Google DeepMind
        """
        ALPHA = 0.7
        ranks = np.argsort(np.abs(losses)) + np.finfo(float).eps
        p = (1 / ranks) ** ALPHA
        sample_probabilities = p / np.sum(p)
        return sample_probabilities

    def learn(self, actor_model, target_actor, critic_model, target_critic, actor_optimizer, critic_optimizer):
        # TODO: multi-step learning
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Uniform mini-batch sampling:
        # batch_indices = np.random.choice(record_range, self.batch_size)

        buffer_indices = range(min(self.buffer_counter, self.buffer_capacity))
        critic_losses = self.calc_critic_loss(buffer_indices, target_actor, critic_model, target_critic)
        buffer_critic_loss = np.mean(critic_losses)
        sample_probabilities = self.calc_sample_probabilities(critic_losses)
        batch_indices = np.random.choice(record_range, self.batch_size, p=sample_probabilities.flatten())
        with tf.GradientTape() as tape:
            critic_losses = self.calc_critic_loss(batch_indices, target_actor, critic_model, target_critic)
            critic_loss = tf.math.reduce_mean(tf.gather(critic_losses, batch_indices))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            # action_mean_l2 = tf.math.reduce_mean(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(actions), 1)), 0)
            actor_loss = -tf.math.reduce_mean(critic_value)
            # + ACTION_L2_REG_FACTOR * action_mean_l2
            actor_loss = tf.identity(actor_loss)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

        return actor_loss, buffer_critic_loss


def get_actor(state_size, action_size):
    inputs = layers.Input(shape=(state_size,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(128, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(16, activation="relu")(out)
    out = layers.BatchNormalization()(out)

    # out = layers.Dense(action_size * 3)(out)
    # out = layers.Reshape((-1, action_size, 3))(out)
    # out = activations.softmax(out, axis=2)

    outputs = layers.Dense(action_size, activation="tanh")(out)
    # outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    # outputs = layers.Dense(action_size, activation="tanh", activity_regularizer='l2')(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(state_size, action_size):
    state_input = layers.Input(shape=(state_size))
    state_out = layers.Dense(32, activation="relu")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(64, activation="relu")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32, activation="relu")(state_out)
    state_out = layers.BatchNormalization()(state_out)

    action_input = layers.Input(shape=(action_size))
    action_out = layers.Dense(16, activation="relu")(action_input)
    action_out = layers.BatchNormalization()(action_out)
    action_out = layers.Dense(32, activation="relu")(action_input)
    action_out = layers.BatchNormalization()(action_out)
    # action_out = layers.Dense(16, activation="relu")(action_input)
    # action_out = layers.BatchNormalization()(action_out)
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(32, activation="relu")(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(16, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def get_critic_value(critic_model, state, action):
    state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    action = tf.expand_dims(tf.convert_to_tensor(action), 0)
    return critic_model([state, action])[0][0]
