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
logging.basicConfig(format='%(asctime)-15s [%(levelname)s]: %(message)s', level=logging.INFO)


class DDPG:
    GAMMA = 0.99
    TAU = 0.05
    CRITIC_LR = 0.002
    ACTOR_LR = 0.001
    MAX_EPISODES = 10000000
    BUFFER_SIZE = 2**15
    BATCH_SIZE = 2**10
    NO_NOISE_TEST_EPISODES = 3
    MAX_NOISE_LEVEL = 0.1

    def __init__(self):
        self.env = Environment(Shape.Worm())
        self.dirname = os.path.dirname(__file__)
        self.best_run = 0
        self.addative_noise_generator = policy_gradient.OUActionNoise(output_size=self.env.action_size)
        self.multiplier_noise_generator = policy_gradient.MarkovSaltPepperNoise(output_size=self.env.action_size)

        self.actor_model = policy_gradient.get_actor(self.env.state_size, self.env.action_size)
        self.critic_model = policy_gradient.get_critic(self.env.state_size, self.env.action_size)
        self.target_actor = policy_gradient.get_actor(self.env.state_size, self.env.action_size)
        self.target_critic = policy_gradient.get_critic(self.env.state_size, self.env.action_size)

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.CRITIC_LR)
        self.actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.ACTOR_LR)
        self.buffer = policy_gradient.Buffer(self.env.state_size, self.env.action_size,
                                             self.GAMMA, self.BUFFER_SIZE, self.BATCH_SIZE)

        self.episode_reward_history = []
        self.show = False
        self.learn = True

    def update_target(self):
        new_weights = []
        target_variables = self.target_critic.weights
        for i, variable in enumerate(self.critic_model.weights):
            new_weights.append(variable * self.TAU + target_variables[i] * (1 - self.TAU))

        self.target_critic.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor.weights
        for i, variable in enumerate(self.actor_model.weights):
            new_weights.append(variable * self.TAU + target_variables[i] * (1 - self.TAU))

        self.target_actor.set_weights(new_weights)

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noised_sampled_actions = sampled_actions

        if self.learn:
            addative_noise = self.addative_noise_generator()
            multiplier_noise = self.multiplier_noise_generator()
            noised_sampled_actions += addative_noise
            noised_sampled_actions *= multiplier_noise
            logging.debug('action {}, noise mul: {}, noise add: {}, total: {}'.format(sampled_actions,
                                                                                      multiplier_noise, addative_noise,
                                                                                      noised_sampled_actions))

        legal_action = np.clip(noised_sampled_actions, -1, 1)
        return np.squeeze(legal_action)

    def apply_keyboard_input_on_action(self, action):
        for i in range(min(len(action), 10)):
            if keyboard.is_pressed(str(i)):
                logging.debug('keyboard input detected')
                if keyboard.is_pressed('up arrow'):
                    action[i] = 1
                if keyboard.is_pressed('down arrow'):
                    action[i] = -1
        if keyboard.is_pressed('q'):
            raise Exception('Keyboard Quit')
        return action

    def episode(self, learn, episode_index):
        prev_state = self.env.reset()
        total_episode_reward = 0
        reward = 0
        done = False
        while not done:
            prev_state_tensor = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = self.policy(prev_state_tensor)
            action = self.process_keyboard(action)
            state, reward, done, info = self.env.step(action)
            self.buffer.record((prev_state, action, reward, state))
            total_episode_reward += reward

            if self.learn:
                actor_loss, critic_loss = self.buffer.learn(self.actor_model, self.target_actor, self.critic_model,
                                                            self.target_critic, self.actor_optimizer,
                                                            self.critic_optimizer)
                self.update_target()
            else:
                critic_loss = 0
                actor_loss = 0
            critic_value = policy_gradient.get_critic_value(self.critic_model, state, action)
            debug_string = '\n'.join((
                "Episode: {} [{}]".format(episode_index, self.env.step_index),
                'Episode Reward: {:0.1f} [{:0.1f}]'.format(total_episode_reward, reward),
                'Episode Distance: {:0.1f}'.format(self.env.get_score()),
                'Velocity: {:0.1f}'.format(self.env.get_walker_x_velocity()),
                'Critic Value: {:0.1f}'.format(critic_value),
                'Action: ' + ', '.join(['{:+0.1f}'.format(i) for i in action]),
            ))
            if self.learn:
                debug_string += '\nCritic Loss: {:0.1f}'.format(critic_loss)
                debug_string += '\nActor Loss: {:0.1f}'.format(actor_loss)
            logging.debug(debug_string)
            if self.show:
                self.render(debug_string)
            prev_state = state
        return total_episode_reward, self.env.step_index

    def render(self, debug_string):
        self.env.render()
        self.env.display.debug_screen_print(debug_string)

    def process_keyboard(self, action):
        if keyboard.is_pressed('d'):
            logging.getLogger().setLevel(logging.DEBUG)
        if keyboard.is_pressed('i'):
            logging.getLogger().setLevel(logging.INFO)

        if keyboard.is_pressed('l'):
            self.learn = True
        if keyboard.is_pressed('k'):
            self.learn = False

        if keyboard.is_pressed('o'):
            self.load_models()

        if keyboard.is_pressed('s'):
            if not self.show:
                self.env.open_window()
            self.show = True
        if keyboard.is_pressed('a'):
            if self.show:
                self.env.close_window()
            self.show = False
        action = self.apply_keyboard_input_on_action(action)
        return action

    def run(self):
        # Takes about 20 min to train
        for episode_index in range(self.MAX_EPISODES):
            start_time = time.time()
            total_episode_reward, steps = self.episode(self.learn, episode_index)

            self.episode_reward_history.append(total_episode_reward)
            pace = (time.time() - start_time) / steps
            average_reward = np.mean(self.episode_reward_history[-10:])
            logging.info("Episode {}: Steps: {} [{:0.2f} sec/step] Avg Reward: {:0.1f}".format(episode_index,
                                                                                               steps, pace,
                                                                                               average_reward))
            if episode_index % 10 == 0:

                average_reward_test = np.mean([self.episode(learn=False, episode_index=episode_index)[0]
                                               for i in range(self.NO_NOISE_TEST_EPISODES)])

                logging.info("Test Episodes {}: Avg Reward: {:0.1f}".format(episode_index,
                                                                            average_reward_test))
                noise_level = min(self.MAX_NOISE_LEVEL, 1 / max(np.finfo(float).eps, average_reward_test))
                self.addative_noise_generator = policy_gradient.OUActionNoise(
                    output_size=self.env.action_size, std_deviation=50 * noise_level)
                self.multiplier_noise_generator = policy_gradient.MarkovSaltPepperNoise(
                    output_size=self.env.action_size, salt_to_pepper=noise_level)

                if average_reward_test > self.best_run:
                    self.best_run = average_reward_test
                    self.save_models()

    def load_models(self):
        logging.info("Weights loaded from {}".format(os.path.join(self.dirname, 'outputs')))
        self.actor_model.load_weights(os.path.join(self.dirname, "outputs\\walker_actor.h5"))
        self.critic_model.load_weights(os.path.join(self.dirname, "outputs\\walker_critic.h5"))
        self.target_actor.load_weights(os.path.join(self.dirname, "outputs\\walker_target_actor.h5"))
        self.target_critic.load_weights(os.path.join(self.dirname, "outputs\\walker_target_critic.h5"))

    def save_models(self):
        logging.info("Weights saved to {}".format(os.path.join(self.dirname, 'outputs')))
        self.actor_model.save_weights(os.path.join(self.dirname, "outputs\\walker_actor.h5"))
        self.critic_model.save_weights(os.path.join(self.dirname, "outputs\\walker_critic.h5"))
        self.target_actor.save_weights(os.path.join(self.dirname, "outputs\\walker_target_actor.h5"))
        self.target_critic.save_weights(os.path.join(self.dirname, "outputs\\walker_target_critic.h5"))


if __name__ == '__main__':
    DDPG().run()
