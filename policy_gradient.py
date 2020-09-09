import numpy as np
import tensorflow as tf

# import matplotlib.pyplot as plt

from tensorflow.keras import layers
# from tensorflow.keras import activations

UPPER_BOUND = 1
LOWER_BOUND = -1
ACTION_REGULARIZATION = 0
ACTION_L2_REG_FACTOR = 0
CRITIC_L2_REG_FACTOR = 1


def calc_critic_loss(target_actor, critic_model, target_critic, gamma,
                     state_batch, action_batch, reward_batch, next_state_batch):
    target_actions = target_actor(next_state_batch)
    target_critic_values = target_critic([next_state_batch, target_actions])
    critic_values = critic_model([state_batch, action_batch])
    gamma_tf = tf.constant(gamma, dtype=tf.float64)
    critic_regularization_factor_tf = tf.constant(CRITIC_L2_REG_FACTOR, dtype=tf.float64)
    critic_losses = critic_loss_tf_function(reward_batch, target_critic_values, gamma_tf,
                                            critic_values, critic_regularization_factor_tf)
    return critic_losses


@tf.function
def critic_loss_tf_function(reward_batch, target_critic, gamma, critic_values, critic_regularization_factor_tf):
    y = reward_batch + gamma * target_critic
    critic_losses = tf.math.square(y - critic_values) + critic_regularization_factor_tf * tf.math.square(critic_values)
    return critic_losses


def get_actor(state_size, action_size):
    inputs = layers.Input(shape=(state_size,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(128, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(128, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.BatchNormalization()(out)

    # outputs = layers.Dense(action_size, activation="tanh")(out)
    # outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    outputs = layers.Dense(action_size, activation="tanh", activity_regularizer='l2')(out)

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
