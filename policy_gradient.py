import logging
import tensorflow as tf
import keyboard
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers

GAMMA = 0.50
# TAU = 0.05
UPPER_BOUND = 1
LOWER_BOUND = -1
ACTION_REGULARIZATION = 0


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.5, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        # x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + (
        #     self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)))
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * self.dt * np.random.normal(size=self.mean.shape)
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class MarkovSaltPepperNoise:
    def __init__(self, shape, salt_to_pepper=0.02, pepper_to_salt=0.1):
        self.salt_to_pepper = salt_to_pepper
        self.pepper_to_salt = pepper_to_salt
        self.noise = np.ones(shape)

    def _reward_to_probabilty(self, reward):
        MAX_PROB = 0.04
        MIN_PROB = 0.001
        MIN_REWARD = 0
        MAX_REWARD = 5
        reward = min(MAX_REWARD, max(reward, MIN_REWARD))
        reward_streched = (reward - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)
        return MIN_PROB + (1 - reward_streched) * (MAX_PROB - MIN_PROB)

    def __call__(self, reward):
        self.salt_to_pepper = self._reward_to_probabilty(reward)
        switch_salt_to_pepper = np.random.binomial(1, self.salt_to_pepper, self.noise.shape)
        switch_pepper_to_salt = np.random.binomial(1, self.pepper_to_salt, self.noise.shape)
        self.noise[(self.noise == 1) & (switch_salt_to_pepper == 1)] = -1
        self.noise[(self.noise == -1) & (switch_pepper_to_salt == 1)] = 1
        return self.noise


class Buffer:
    def __init__(self, state_size, action_size, buffer_capacity=100000, batch_size=64):
        # plt.ion()

        # self.im = plt.imshow(np.zeros((77, 36)), cmap='gray', vmin=-0.5, vmax=0.5)

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_size))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self, actor_model, critic_model, actor_optimizer, critic_optimizer):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        pick_probability = self.reward_buffer[0:record_range] - np.min(self.reward_buffer[0:record_range]) \
            + np.finfo(float).eps
        pick_probability /= np.sum(pick_probability)
        batch_indices = np.random.choice(record_range, self.batch_size, p=pick_probability.reshape(-1))

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            # target_actions = target_actor(next_state_batch)
            target_actions = actor_model(next_state_batch)
            if np.isnan(np.sum(target_actions.numpy())):
                return
            # y = reward_batch + GAMMA * target_critic([next_state_batch, target_actions])
            y = reward_batch + GAMMA * critic_model([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        assert not np.isnan(np.sum(action_batch.numpy()))
        assert not np.isnan(np.sum(state_batch.numpy()))
        assert not np.isnan(np.sum(next_state_batch.numpy()))
        assert not np.isnan(np.sum(reward_batch.numpy()))
        assert not np.isnan(np.sum(target_actions.numpy()))
        assert not np.isnan(np.sum(critic_value.numpy()))
        assert not np.isnan(np.sum(critic_loss.numpy()))

        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        # im.set_array(np.hstack([a.numpy().flatten() for a in actor_model.trainable_variables]).reshape((77, 36)))
        # disp_mat = np.hstack([a.numpy().flatten() for a in actor_grad]).reshape((77, 36))
        # self.im.set_array(disp_mat)
        # plt.draw()

        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )


# This update target parameters slowly
# Based on rate `TAU`, which is much less than one.
# def update_target():
#     new_weights = []
#     target_variables = target_critic.weights
#     for i, variable in enumerate(critic_model.weights):
#         new_weights.append(variable * TAU + target_variables[i] * (1 - TAU))

#     target_critic.set_weights(new_weights)

#     new_weights = []
#     target_variables = target_actor.weights
#     for i, variable in enumerate(actor_model.weights):
#         new_weights.append(variable * TAU + target_variables[i] * (1 - TAU))

#     target_actor.set_weights(new_weights)


def get_actor(state_size, action_size):
    # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(state_size,))
    out = layers.Dense(128, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(64, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(32, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(32, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(16, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(out)
    out = layers.BatchNormalization()(out)
    # outputs = layers.Dense(action_size, kernel_regularizer='l2', bias_regularizer='l2')(out)
    outputs = layers.Dense(action_size, activation="tanh", kernel_regularizer='l2', bias_regularizer='l2')(out)
    # outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    # outputs = layers.Dense(action_size, activation="tanh", activity_regularizer='l2')(out)

    # outputs = outputs * UPPER_BOUND
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(state_size, action_size):
    # State as input
    state_input = layers.Input(shape=(state_size))
    state_out = layers.Dense(128, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(64, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(state_out)
    state_out = layers.BatchNormalization()(state_out)

    # Action as input
    action_input = layers.Input(shape=(action_size))
    action_out = layers.Dense(32, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(action_input)
    action_out = layers.BatchNormalization()(action_out)
    action_out = layers.Dense(16, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(action_input)
    action_out = layers.BatchNormalization()(action_out)
    action_out = layers.Dense(16, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(action_input)
    action_out = layers.BatchNormalization()(action_out)
    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(32, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(16, activation="relu", kernel_regularizer='l2', bias_regularizer='l2')(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def apply_keyboard_input(action):
    for i in range(min(len(action), 9)):
        if keyboard.is_pressed(str(i)):
            logging.debug('keyboard input detected')
            if keyboard.is_pressed('up arrow'):
                action[i] = 1
            if keyboard.is_pressed('down arrow'):
                action[i] = -1
    if keyboard.is_pressed('q'):
        raise Exception('Keyboard Quit')
    return action


# action_history = []


def policy(state, reward, multiplier_noise_generator, addative_noise_generator, actor_model):
    # SWITCH_THRESHOLD = 0.9
    # global action_history
    sampled_actions = tf.squeeze(actor_model(state)).numpy()

    # action_history.pop(0)
    # action_history.append(sampled_actions)
    # sampled_actions(np.abs(np.mean(action_history)) > SWITCH_THRESHOLD)
    if np.isnan(np.sum(sampled_actions)):
        return np.zeros(sampled_actions.shape)
    multiplier_noise = multiplier_noise_generator(reward)
    addative_noise = addative_noise_generator()

    noised_sampled_actions = (sampled_actions + addative_noise) * multiplier_noise
    noised_sampled_actions = apply_keyboard_input(noised_sampled_actions)
    legal_action = np.clip(noised_sampled_actions, LOWER_BOUND, UPPER_BOUND)
    # logging.info('action {}, noise mul: {}, noise add: {}, total: {}'.format(sampled_actions,
    #                                                                          multiplier_noise, addative_noise,
    #                                                                          noised_sampled_actions))
    assert not np.isnan(np.sum(state))
    assert not np.isnan(np.sum(addative_noise))
    assert not np.isnan(np.sum(multiplier_noise))
    assert not np.isnan(np.sum(sampled_actions))
    assert not np.isnan(np.sum(noised_sampled_actions))
    assert not np.isnan(np.sum(legal_action))
    return np.squeeze(legal_action)


def get_critic_value(critic_model, state, action):
    state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    action = tf.expand_dims(tf.convert_to_tensor(action), 0)
    return critic_model([state, action]).numpy()[0][0]
