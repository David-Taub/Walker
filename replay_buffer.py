import policy_gradient
import tensorflow as tf
import logging
import numpy as np
import mlflow


class PrioritizedBuffer:
    def __init__(self, state_size, action_size, gamma, buffer_capacity=100000, batch_size=64):

        # self.im = plt.imshow(np.zeros((77, 36)), cmap='gray', vmin=-0.5, vmax=0.5)
        self.gamma = gamma
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_current_size = 0
        self.buffer_write_index = 0
        self.state_buffer = np.zeros((self.buffer_capacity, state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_size))

    def record(self, observation):
        # when the buffer is not in full capacity, we fill it from the top
        # when it is full, we overwrite from the end toward the beginning
        index = self.buffer_write_index if self.buffer_write_index < self.buffer_capacity else \
            self.buffer_capacity - 1 - (self.buffer_write_index % self.buffer_capacity)

        self.state_buffer[index] = observation[0]
        self.action_buffer[index] = observation[1]
        self.reward_buffer[index] = observation[2]
        self.next_state_buffer[index] = observation[3]

        self.buffer_write_index += 1
        self.buffer_current_size = min(self.buffer_capacity, self.buffer_current_size + 1)
        logging.debug('Writing in buffer at {}'.format(index))

    def get_prioritize_batch_indices(self):
        """
        Rank variant of "Prioritized Experience Replay", Schaul et al. 2015 of Google DeepMind
        """
        ALPHA = 0.5
        ranks = np.arange(1, self.buffer_current_size + 1)
        p = (1 / ranks) ** ALPHA
        sample_probabilities = p / np.sum(p)
        batch_indices = np.random.choice(self.buffer_current_size, self.batch_size, p=sample_probabilities.flatten())
        return batch_indices

    def prioritize_buffer(self, target_actor, critic_model, target_critic):
        logging.debug('Prioritizing buffer with {} records'.format(self.buffer_current_size))
        self.buffer_write_index = self.buffer_current_size
        buffer_indices = range(self.buffer_current_size)
        critic_losses = policy_gradient.calc_critic_loss(target_actor, critic_model, target_critic, self.gamma,
                                                         self.state_buffer[buffer_indices],
                                                         self.action_buffer[buffer_indices],
                                                         self.reward_buffer[buffer_indices],
                                                         self.next_state_buffer[buffer_indices])
        mlflow.log_metric('buffer_critic_loss_mean', tf.math.reduce_mean(critic_losses).numpy())
        mlflow.log_metric('buffer_critic_loss_std', tf.math.reduce_std(critic_losses).numpy())
        sorted_indices = np.argsort(np.abs(critic_losses.numpy().flatten()))[::-1]
        self.state_buffer[buffer_indices] = self.state_buffer[sorted_indices]
        self.action_buffer[buffer_indices] = self.action_buffer[sorted_indices]
        self.reward_buffer[buffer_indices] = self.reward_buffer[sorted_indices]
        self.next_state_buffer[buffer_indices] = self.next_state_buffer[sorted_indices]

    def learn(self, actor_model, target_actor, critic_model, target_critic, actor_optimizer, critic_optimizer):
        # TODO: multi-step learning

        # Uniform mini-batch sampling:
        # batch_indices = np.random.choice(min(self.buffer_write_index, self.buffer_capacity), self.batch_size)
        batch_indices = self.get_prioritize_batch_indices()
        with tf.GradientTape() as tape:
            critic_losses = policy_gradient.calc_critic_loss(target_actor, critic_model, target_critic, self.gamma,
                                                             tf.gather(self.state_buffer * 1, batch_indices),
                                                             tf.gather(self.action_buffer * 1, batch_indices),
                                                             tf.gather(self.reward_buffer * 1, batch_indices),
                                                             tf.gather(self.next_state_buffer * 1, batch_indices))
            critic_loss = tf.math.reduce_mean(critic_losses)
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
            # action_mean_l2 = tf.math.reduce_mean(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(actions), 1)), 0)
            # + ACTION_L2_REG_FACTOR * action_mean_l2

        mlflow.log_metric('batch_actor_loss', actor_loss.numpy())
        mlflow.log_metric('batch_critic_loss', critic_loss.numpy())
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

        return actor_loss, critic_loss
