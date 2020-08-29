# original source:
# https://raw.githubusercontent.com/keras-team/keras-io/master/examples/rl/ddpg_pendulum.py
import logging
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard
import utils
from Environment import Environment
import policy_gradient
import Shape

np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
env = Environment(Shape.Worm())
# env = Environment(Shape.Legs())

state_size = env.state_size
action_size = env.action_size

GAMMA = 0.99
TAU = 0.05
CRITIC_LR = 0.002
ACTOR_LR = 0.001
MAX_STEPS_PER_EPISODE = 200
MAX_EPISODES = 10000000
BUFFER_SIZE = 1000
BATCH_SIZE = 4096

addative_noise_generator = policy_gradient.OUActionNoise(mean=np.zeros(
    action_size), std_deviation=1 * np.ones(action_size), theta=0.5, dt=0.01)
multiplier_noise_generator = policy_gradient.MarkovSaltPepperNoise(shape=(action_size,))

actor_model = policy_gradient.get_actor(state_size, action_size)
critic_model = policy_gradient.get_critic(state_size, action_size)
target_actor = policy_gradient.get_actor(state_size, action_size)
target_critic = policy_gradient.get_critic(state_size, action_size)

target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models

critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=CRITIC_LR)
actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=ACTOR_LR)

# Discount factor for future rewards
# Used to update target networks

buffer = policy_gradient.Buffer(state_size, action_size, GAMMA, BUFFER_SIZE, BATCH_SIZE)

tf.keras.utils.plot_model(actor_model, show_shapes=True)
tf.keras.utils.plot_model(critic_model, show_shapes=True)


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
show = False
learn = True

# Takes about 20 min to train
for episode_index in range(MAX_EPISODES):
    prev_state = env.reset()
    episodic_reward = 0
    reward = 0
    start_time = time.time()
    for step_index in range(MAX_STEPS_PER_EPISODE):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        # utils.tic()
        if learn:
            action = policy_gradient.policy(tf_prev_state, reward, multiplier_noise_generator,
                                            addative_noise_generator, actor_model)
        else:
            action = policy_gradient.policy(tf_prev_state, reward, None,
                                            None, actor_model)
        # utils.toc('policy')
        # Recieve state and reward from environment.
        # utils.tic()
        state, reward, done, info = env.step(action)
        # utils.toc('step')
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        # utils.tic()
        if keyboard.is_pressed('l'):
            learn = True
        if keyboard.is_pressed('k'):
            learn = False
        if learn:
            actor_loss, critic_loss = buffer.learn(actor_model, target_actor, critic_model,
                                                   target_critic, actor_optimizer, critic_optimizer)
            # utils.toc('learn')

            policy_gradient.update_target(actor_model, target_actor, critic_model, target_critic, TAU)

        # env.render() if step_index % 5 == 0 else None
        # utils.tic()
        if keyboard.is_pressed('s'):
            show = True
        if keyboard.is_pressed('a'):
            show = False
        if show:
            env.render()
            env.display.debug_screen_print('\n'.join((
                "Episode: {} [{}]".format(episode_index, step_index),
                'Episode Reward: {:0.1f} [{:0.1f}]'.format(episodic_reward, reward),
                'Episode Distance: {:0.1f}'.format(env.get_score()),
                'Velocity: {:0.1f}'.format(env.get_walker_x_velocity()),
                'Critic Value: {:0.1f}'.format(policy_gradient.get_critic_value(critic_model, state, action)),
                'Critic Loss: {:0.1f}'.format(critic_loss),
                'Action: ' + ', '.join(['{:+0.1f}'.format(i) for i in action]),
                'Actor Loss: {:0.1f}'.format(actor_loss)
            )))
        # utils.toc('render')
        # End this episode when `done` is True
        # if done:
        if done and step_index > 30:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    pace = (time.time() - start_time) / step_index
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode {}: Steps: {} [{:0.2f} sec/step] Avg Reward: {:0.1f}".format(episode_index,
                                                                                step_index, pace, avg_reward))
    avg_reward_list.append(avg_reward)
    # for i in range(5):
    #     buffer.learn(actor_model, critic_model, actor_optimizer, critic_optimizer)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()


# Save the weights
# actor_model.save_weights("outputs\\walker_actor.h5")
# critic_model.save_weights("outputs\\walker_critic.h5")

target_actor.save_weights("outputs\\walker_target_actor.h5")
target_critic.save_weights("outputs\\walker_target_critic.h5")
