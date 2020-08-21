# original source:
# https://raw.githubusercontent.com/keras-team/keras-io/master/examples/rl/ddpg_pendulum.py
import logging
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from Environment import Environment
import policy_gradient

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
tf.enable_eager_execution()

env = Environment()

state_size = env.state_size
action_size = env.action_size
STD_DEV = 0.2
CRITIC_LR = 0.0001
ACTOR_LR = 0.0001
MAX_STEPS_PER_EPISODE = 100
MAX_EPISODES = 10000
BUFFER_SIZE = 1000
BATCH_SIZE = 128


ou_noise = policy_gradient.OUActionNoise(mean=np.zeros(
    action_size), std_deviation=float(STD_DEV) * np.ones(action_size))

actor_model = policy_gradient.get_actor(state_size, action_size)
critic_model = policy_gradient.get_critic(state_size, action_size)

# target_actor = get_actor()
# target_critic = get_critic()

# Making the weights equal initially
# target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models

critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)
actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)

# Discount factor for future rewards
# Used to update target networks

buffer = policy_gradient.Buffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE)

tf.keras.utils.plot_model(actor_model, show_shapes=True)
tf.keras.utils.plot_model(critic_model, show_shapes=True)


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


# Takes about 20 min to train
for episode_index in range(MAX_EPISODES):

    prev_state = env.reset()
    episodic_reward = 0

    for step_index in range(MAX_STEPS_PER_EPISODE):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy_gradient.policy(tf_prev_state, ou_noise, actor_model)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn(actor_model, critic_model, actor_optimizer, critic_optimizer)

        # update_target(TAU)

        env.render()
        env.display.debug_screen_print('\n'.join((
            "Episode: {} [{}]".format(episode_index, step_index),
            'Episode Reward: {:0.1f}'.format(episodic_reward),
            'Critic Value: {:0.2f}'.format(policy_gradient.get_critic_value(critic_model, state, action)),
            'Score: {:0.1f}'.format(env.get_score()),
            'Action: ' + ', '.join(['{:+0.1f}'.format(i) for i in action]),
            # 'State: ' + ', '.join(['{:0.1f}'.format(i) for i in state]),
        )))
        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(episode_index, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()


# Save the weights
actor_model.save_weights("walker_actor.h5")
critic_model.save_weights("walker_critic.h5")

# target_actor.save_weights("walker_target_actor.h5")
# target_critic.save_weights("walker_target_critic.h5")
