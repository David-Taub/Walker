import logging
import time

import keyboard
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow as tf

from Environment import Environment

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
tf.enable_eager_execution()


def apply_keyboard_input(action):
    for i in range(min(len(action), 9)):
        if keyboard.is_pressed(str(i)):
            logging.debug('keyboard input detected')
            if keyboard.is_pressed('up arrow'):
                action[i] = 1
            if keyboard.is_pressed('down arrow'):
                action[i] = -1
    return action


def tic():
    global t
    t = time.time()


def toc(s):
    logging.debug('{0} time: {1:0.2f}'.format(s, time.time() - t))


env = Environment()

state_size = len(env.get_current_state())
num_actions = env.joints_count
num_hidden = 128
inputs = layers.Input(shape=(state_size,))
hidden_layer = layers.Dense(num_hidden, activation="relu")(inputs)
action_layer = layers.Dense(num_actions, activation="sigmoid")(hidden_layer)
critic_layer = layers.Dense(1)(hidden_layer)
model = keras.Model(inputs=inputs, outputs=[action_layer, critic_layer])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
MAX_STEPS_PER_EPISODE = 1000
MAX_EPISODES = 1000000
EPISODES_INTERVAL_TO_RENDER = 1
GAMMA = 0.5

for episode_index in range(MAX_EPISODES):
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, MAX_STEPS_PER_EPISODE):
            if episode_index % EPISODES_INTERVAL_TO_RENDER == 0:
                tic()
                # env.render()
                toc('render')
            tic()
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action, critic_value = model(state)
            toc('model')
            tic()
            critic_value_history.append(critic_value[0, 0])
            # TODO: eval makes the model slower ans slower. should use eager mode indsta
            # action_vec = K.eval(action)[0]
            action_vec = action.numpy()[0]
            # for debugging, keyboard input can affect the action
            action_vec = apply_keyboard_input(action_vec)

            # Sample action from action probability distribution
            action_history.append(action)
            toc('action')

            # Apply the sampled action in our environment
            tic()
            state, reward, done, _ = env.step(action_vec)
            toc('step')
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with GAMMA
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + np.finfo(float).eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        logging.info('train')
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        logging.info(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        logging.info("Solved at episode {}!".format(episode_count))
        break
