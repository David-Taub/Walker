# original source:
# https://raw.githubusercontent.com/keras-team/keras-io/master/examples/rl/ddpg_pendulum.py
import logging
import os
import tensorflow as tf

import numpy as np
import time
import keyboard
from Environment import Environment
import policy_gradient
import noise_generators
from replay_buffer import PrioritizedBuffer
import Shape
import mlflow

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
    BUFFER_SIZE = 10000
    BATCH_SIZE = 512
    NO_NOISE_TEST_EPISODES = 3
    MAX_NOISE_LEVEL = 0.1

    def __init__(self):

        self.env = Environment(Shape.Worm())
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), 'mlflow')
        self.best_run_multiple_episodes = 0
        self.addative_noise_generator = noise_generators.OUActionNoise(output_size=self.env.action_size)
        self.multiplier_noise_generator = noise_generators.MarkovSaltPepperNoise(output_size=self.env.action_size)

        self.init_models()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.CRITIC_LR)
        self.actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.ACTOR_LR)
        self.buffer = PrioritizedBuffer(self.env.state_size, self.env.action_size,
                                        self.GAMMA, self.BUFFER_SIZE, self.BATCH_SIZE)

        self.episode_reward_history = []
        # show controls the appearance of a window with graphics, controlled by 's' and 'a' on the keyboard
        self.show = False
        # show controls if the model is learning or not, affects the FPS of the graphics
        # controlled by 'l' and 'k' on the keyboard
        self.learn = True
        self.log_params()

    def init_models(self):
        self.actor_model = policy_gradient.get_actor(self.env.state_size, self.env.action_size)
        self.critic_model = policy_gradient.get_critic(self.env.state_size, self.env.action_size)
        self.target_actor = policy_gradient.get_actor(self.env.state_size, self.env.action_size)
        self.target_critic = policy_gradient.get_critic(self.env.state_size, self.env.action_size)

    def log_params(self):
        mlflow.log_param('GAMMA', self.GAMMA)
        mlflow.log_param('TAU', self.TAU)
        mlflow.log_param('CRITIC_LR', self.CRITIC_LR)
        mlflow.log_param('ACTOR_LR', self.ACTOR_LR)
        mlflow.log_param('MAX_EPISODES', self.MAX_EPISODES)
        mlflow.log_param('BUFFER_SIZE', self.BUFFER_SIZE)
        mlflow.log_param('BATCH_SIZE', self.BATCH_SIZE)
        mlflow.log_param('NO_NOISE_TEST_EPISODES', self.NO_NOISE_TEST_EPISODES)
        mlflow.log_param('MAX_NOISE_LEVEL', self.MAX_NOISE_LEVEL)

        mlflow.log_param('JOINT_POWER', self.env.JOINT_POWER)
        mlflow.log_param('JOINT_SPEED', self.env.JOINT_SPEED)
        mlflow.log_param('PLANE_FRICTION', self.env.PLANE_FRICTION)
        mlflow.log_param('GRAVITY_ACCELERATION', self.env.GRAVITY_ACCELERATION)
        mlflow.log_param('ANGLE_SCALE', self.env.ANGLE_SCALE)
        mlflow.log_param('PHYSICAL_STEPS_PER_ACTION', self.env.PHYSICAL_STEPS_PER_ACTION)
        mlflow.log_param('MAX_STEPS_PER_EPISODE', self.env.MAX_STEPS_PER_EPISODE)
        mlflow.log_param('MAX_STABILITY_STEPS', self.env.MAX_STABILITY_STEPS)
        mlflow.log_param('MIN_MOVEMENT_FOR_STABILITY', self.env.MIN_MOVEMENT_FOR_STABILITY)
        mlflow.log_param('LAST_VELOCITY_HISTORY_SIZE', self.env.LAST_VELOCITY_HISTORY_SIZE)
        mlflow.log_param('LAST_VELOCITY_AVERAGE_INIT', self.env.LAST_VELOCITY_AVERAGE_INIT)
        mlflow.log_param('MIN_MOVEMENT_FOR_END_EPISODE', self.env.MIN_MOVEMENT_FOR_END_EPISODE)
        mlflow.log_param('TIME_STEP_REWARD', self.env.TIME_STEP_REWARD)
        mlflow.log_param('VELOCITY_REWARD', self.env.VELOCITY_REWARD)
        mlflow.log_param('VELOCITY_DECREASE_PENALTY', self.env.VELOCITY_DECREASE_PENALTY)
        mlflow.log_param('SIDE_PROGRESS_PENALTY', self.env.SIDE_PROGRESS_PENALTY)
        mlflow.log_param('ACTUATOR_PENALTY', self.env.ACTUATOR_PENALTY)
        mlflow.log_param('OVER_PRESS_JOINT_PENALTY', self.env.OVER_PRESS_JOINT_PENALTY)

    def get_critic_value(self, state, action):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = tf.expand_dims(tf.convert_to_tensor(action), 0)
        return self.critic_model([state, action])[0][0]

    def update_target_models(self):
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
        # for i, val in enumerate(sampled_actions):
        #     mlflow.log_metric('action_unnoised_{}'.format(i), val.numpy())
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
        # for i, val in enumerate(legal_action):
        #     mlflow.log_metric('action_noised_{}'.format(i), val)
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
                self.update_target_models()
            else:
                critic_loss = 0
                actor_loss = 0
            critic_value = self.get_critic_value(state, action)
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

    def run_multiple_episodes(self):
        for episode_index in range(self.MAX_EPISODES):
            start_time = time.time()
            total_episode_reward, steps = self.episode(self.learn, episode_index)

            self.episode_reward_history.append(total_episode_reward)
            pace = (time.time() - start_time) / steps
            average_reward = np.mean(self.episode_reward_history[-10:])
            logging.info("Episode {}: Steps: {} [{:0.2f} sec/step] Avg Reward: {:0.1f}".format(episode_index,
                                                                                               steps, pace,
                                                                                               average_reward))
            mlflow.log_metric('episode_step_pace', pace)
            mlflow.log_metric('episode_reward', total_episode_reward)
            mlflow.log_metric('episode_reward_smoothed', average_reward)
            mlflow.log_metric('episode_step_count', steps)
            # TODO: use consts here
            if episode_index % 10 == 0:
                self.buffer.prioritize_buffer(self.target_actor, self.critic_model, self.target_critic)
                average_reward_test = np.mean([self.episode(learn=False, episode_index=episode_index)[0]
                                               for i in range(self.NO_NOISE_TEST_EPISODES)])
                self.noise_level = min(self.MAX_NOISE_LEVEL, 500 / max(np.finfo(float).eps, average_reward_test) ** 0.5)
                self.addative_noise_generator = noise_generators.OUActionNoise(
                    output_size=self.env.action_size, std_deviation=90 * self.noise_level)
                self.multiplier_noise_generator = noise_generators.MarkovSaltPepperNoise(
                    output_size=self.env.action_size, salt_to_pepper=self.noise_level)

                mlflow.keras.log_model(self.target_actor, 'target_actor')
                mlflow.keras.log_model(self.target_critic, 'target_critic')
                mlflow.keras.log_model(self.actor_model, 'actor_model')
                mlflow.keras.log_model(self.critic_model, 'critic_model')
                if average_reward_test > self.best_run_multiple_episodes:
                    self.best_run_multiple_episodes = average_reward_test
                    self.save_models()
                else:
                    self.load_models()
                logging.info(
                    "Test Episodes {}: Avg Reward: {:0.1f} (best: {:0.1f})".format(episode_index,
                                                                                   average_reward_test,
                                                                                   self.best_run_multiple_episodes))
                # mlflow.log_metric('episode_noise_level', noise_level)
                mlflow.log_metric('episode_reward_test', average_reward_test)
                mlflow.log_metric('best_run_multiple_episodes', self.best_run_multiple_episodes)

    def load_models(self):
        try:
            self.actor_model = tf.keras.models.load_model(os.path.join(self.checkpoint_dir, 'actor_model'))
            self.critic_model = tf.keras.models.load_model(os.path.join(self.checkpoint_dir, 'critic_model'))
            self.target_actor = tf.keras.models.load_model(os.path.join(self.checkpoint_dir, 'target_actor'))
            self.target_critic = tf.keras.models.load_model(os.path.join(self.checkpoint_dir, 'target_critic'))
            logging.info("Weights loaded from {}".format(self.checkpoint_dir))
        except Exception:
            logging.warning("Weights couldn't be loaded from {}".format(self.checkpoint_dir))
            pass

    def save_models(self):
        tf.keras.models.save_model(self.actor_model, os.path.join(self.checkpoint_dir, 'actor_model'))
        tf.keras.models.save_model(self.critic_model, os.path.join(self.checkpoint_dir, 'critic_model'))
        tf.keras.models.save_model(self.target_actor, os.path.join(self.checkpoint_dir, 'target_actor'))
        tf.keras.models.save_model(self.target_critic, os.path.join(self.checkpoint_dir, 'target_critic'))
        logging.info("Weights saved to {}".format(self.checkpoint_dir))


if __name__ == '__main__':
    with mlflow.start_run_multiple_episodes():
        DDPG().run_multiple_episodes()
