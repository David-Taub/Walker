# original source:
# https://raw.githubusercontent.com/keras-team/keras-io/master/examples/rl/ddpg_pendulum.py
import logging
import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard
from Environment import Environment
import policy_gradient
import Shape

SEED_VALUE = 42
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
tf.keras.backend.set_floatx('float64')
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
env = Environment(Shape.Worm())
# env = Environment(Shape.Legs())

state_size = env.state_size
action_size = env.action_size
best_run = 0
GAMMA = 0.99
TAU = 0.05
MIN_STEPS_IN_EPISODE = 30
CRITIC_LR = 0.002
ACTOR_LR = 0.001
MAX_STEPS_PER_EPISODE = 200
MAX_EPISODES = 10000000
BUFFER_SIZE = 10000
BATCH_SIZE = 1024

addative_noise_generator = policy_gradient.OUActionNoise(mean=np.zeros(
    action_size), std_deviation=2 * np.ones(action_size), theta=0.7, dt=0.01)
multiplier_noise_generator = policy_gradient.MarkovSaltPepperNoise(shape=(action_size,))

actor_model = policy_gradient.get_actor(state_size, action_size)
critic_model = policy_gradient.get_critic(state_size, action_size)
target_actor = policy_gradient.get_actor(state_size, action_size)
target_critic = policy_gradient.get_critic(state_size, action_size)

target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())
critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=CRITIC_LR)
actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=ACTOR_LR)
buffer = policy_gradient.Buffer(state_size, action_size, GAMMA, BUFFER_SIZE, BATCH_SIZE)


episode_reward_history = []
average_reward_list = []
show = False
learn = True


def episode(env, multiplier_noise_generator, addative_noise_generator, learn, show):
    prev_state = env.reset()
    total_episode_reward = 0
    reward = 0
    for step_index in range(MAX_STEPS_PER_EPISODE):
        prev_state_tensor = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        if learn:
            action = policy_gradient.policy(prev_state_tensor, reward, multiplier_noise_generator,
                                            addative_noise_generator, actor_model)
        else:
            action = policy_gradient.policy(prev_state_tensor, reward, None, None, actor_model)
        state, reward, done, info = env.step(action)
        buffer.record((prev_state, action, reward, state))
        total_episode_reward += reward

        if learn:
            actor_loss, critic_loss = buffer.learn(actor_model, target_actor, critic_model,
                                                   target_critic, actor_optimizer, critic_optimizer)
            policy_gradient.update_target(actor_model, target_actor, critic_model, target_critic, TAU)
        else:
            critic_loss = 0
            actor_loss = 0

        if show:
            env.render()
            env.display.debug_screen_print('\n'.join((
                "Episode: {} [{}]".format(episode_index, step_index),
                'Episode Reward: {:0.1f} [{:0.1f}]'.format(total_episode_reward, reward),
                'Episode Distance: {:0.1f}'.format(env.get_score()),
                'Velocity: {:0.1f}'.format(env.get_walker_x_velocity()),
                'Critic Value: {:0.1f}'.format(policy_gradient.get_critic_value(critic_model, state, action)),
                'Critic Loss: {:0.1f}'.format(critic_loss),
                'Action: ' + ', '.join(['{:+0.1f}'.format(i) for i in action]),
                'Actor Loss: {:0.1f}'.format(actor_loss)
            )))
        if done and step_index > MIN_STEPS_IN_EPISODE:
            break

        prev_state = state
    return total_episode_reward, step_index


# Takes about 20 min to train
for episode_index in range(MAX_EPISODES):
    start_time = time.time()
    total_episode_reward, steps = episode(env, multiplier_noise_generator,
                                          addative_noise_generator, learn, show)

    if keyboard.is_pressed('l'):
        learn = True
    if keyboard.is_pressed('k'):
        learn = False

    if keyboard.is_pressed('s'):
        show = True
    if keyboard.is_pressed('a'):
        env.remove_display()
        show = False
    episode_reward_history.append(total_episode_reward)
    pace = (time.time() - start_time) / steps
    average_reward = np.mean(episode_reward_history[-10:])
    logging.info("Episode {}: Steps: {} [{:0.2f} sec/step] Avg Reward: {:0.1f}".format(episode_index,
                                                                                       steps, pace,
                                                                                       average_reward))
    average_reward_list.append(average_reward)
    # if episode_index % 100 == 0:
    #     logging.info("Draw plot")
    #     plt.ion()
    #     plt.plot(average_reward_list)
    #     plt.xlabel("Episode")
    #     plt.ylabel("Avg. Epsiodic Reward")
    #     plt.draw()
    if average_reward > best_run:
        dirname = os.path.dirname(__file__)
        logging.info("Weights saved to {}".format(os.path.join(dirname, 'outputs')))
        best_run = average_reward
        actor_model.save_weights(os.path.join(dirname, "outputs\\walker_actor.h5"))
        critic_model.save_weights(os.path.join(dirname, "outputs\\walker_critic.h5"))
        target_actor.save_weights(os.path.join(dirname, "outputs\\walker_target_actor.h5"))
        target_critic.save_weights(os.path.join(dirname, "outputs\\walker_target_critic.h5"))
