# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

STATE_LAYER_1_FACTOR = 2
STATE_LAYER_2_FACTOR = 1
GENERAL_STATE_LAYER_1_FACTOR = 1
ACTION_LAYER_1_FACTOR = 2
ACTION_LAYER_2_FACTOR = 1
MERGED_LAYER_FATOR_1 = 1.1
MERGED_LAYER_FATOR_2 = 1.1
ACTION_VALUES = [-1, 0, 1]


class QNetwork(object):
    def __init__(self, state_sizes, action_size):
        print('Building Q network')
        self.total_value = 0
        self.LEARNING_RATE = 0.00001
        self.BETA_W = 10 ** -3
        self.weights = []
        self.action_size = action_size
        self.general_state_size = state_sizes[0]
        self.num_of_legs = state_sizes[1]
        self.leg_state_size = state_sizes[2]

        self._init_inputs()
        self._init_legs_layers()
        # self._init_action_layers()
        self._init_general_state_layer()
        self._init_merge_and_q()
        self._init_outputs()

    def _init_outputs(self):
        # advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target_value = tf.placeholder(shape=[None], dtype=tf.float32, name='target_value')
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')
        with tf.name_scope('error'):
            value_loss = tf.reduce_sum(tf.square(self.target_value - self.value))
            policy_loss = -tf.reduce_sum(tf.log(self.q_est)*self.advantages)
            entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
            print(entropy.shape)
            input("4")
            with tf.name_scope('regularization'):
                self.regularizers = sum([tf.nn.l2_loss(weight) for weight in self.weights])

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(value_loss + policy_loss + self.BETA_W * self.regularizers + entropy)
            tf.summary.scalar('total_loss', self.loss)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('policy_loss', policy_loss)
            tf.summary.scalar('regularizer', self.regularizers)
            tf.summary.histogram('advantages', self.advantages)
            tf.summary.histogram('target_value', self.target_value)

        trainer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.update_model = trainer.minimize(self.loss)

    def train(self, discounted_rewards, states, actions, advantages, sess):
        feed_dict = self.states_to_feed_dict(states)
        feed_dict[self.target_value] = discounted_rewards
        feed_dict[self.action_input] = actions
        feed_dict[self.advantages] = advantages
        sess.run(self.update_model, feed_dict=feed_dict)
        return feed_dict

    def get_regularizer(self, sess):
        return sess.run(self.regularizers)

    # get state, predict action by simple per-dimension ascent
    # states - batch_size x state_size
    # actions - batch_size x state_size
    def predict(self, state, sess):
        feed_dict = self.states_to_feed_dict(state)
        policy = sess.run(self.policy, feed_dict=feed_dict)
        action = []
        for i in range(policy.shape[1]):
            action.append(np.random.choice(ACTION_VALUES, 1, p=policy[0,i,:])[0])
            print(policy[0,i,:], action)
        # [aa, qq] = sess.run([self.output_action, self.policy], feed_dict=feed_dict)
        return action

    def get_value(self, states, sess):
        feed_dict = self.states_to_feed_dict(states)
        value = sess.run(self.value, feed_dict=feed_dict)
        self.total_value += value[0]
        return value

    def pop_total_value(self):
        ret = self.total_value
        self.total_value = 0
        return ret

    def states_to_feed_dict(self, states):
        # split_states = np.split(states, np.cumsum(self.state_sizes), 1)[:-1]
        # feed_dict = dict(zip(self.state_inputs, split_states))
        feed_dict = {}
        feed_dict[self.general_state_input] = states[:, :self.general_state_size]
        for i in range(self.num_of_legs):
            start = self.general_state_size + i * self.leg_state_size
            end = self.general_state_size + (i + 1) * self.leg_state_size
            feed_dict[self.legs_states_input[i]] = states[:, start:end]
        return feed_dict

    def _init_inputs(self):
        self.action_input = tf.placeholder(shape=[None, self.action_size], dtype=tf.int8, name="action_input")
        with tf.name_scope('input_state'):
            shape = [None, self.general_state_size]
            self.general_state_input = tf.placeholder(shape=shape, dtype=tf.float32, name="input_general_state")
            self.legs_states_input = []
            shape = [None, self.leg_state_size]
            for i in range(self.num_of_legs):
                self.legs_states_input.append(tf.placeholder(shape=shape, dtype=tf.float32, name="input_general_state"))

    def _init_merge_and_q(self):

        with tf.name_scope('merged_1'):
            to_merge = self.leg_hiddens2 + [self.general_state_hidden1]
            # to_merge = [self.action_hidden1] + self.leg_hiddens2 + [self.general_state_hidden1]
            merged = tf.concat(to_merge, axis=1, name="merged")
            self._variable_summaries(merged, 'merged')
            in_size = merged.get_shape()[1].value
            out_size = round(in_size * MERGED_LAYER_FATOR_1)
            mixed1 = self._gen_layer(merged, in_size, out_size, 'affine1')
            in_size = out_size
            out_size = round(in_size * MERGED_LAYER_FATOR_2)
            mixed2 = self._gen_layer(mixed1, in_size, out_size, 'affine2')
        in_size = out_size
        policy_flat = self._gen_layer(mixed2, in_size, self.action_size * len(ACTION_VALUES), 'policy', False)
        policy_raw = tf.reshape(policy_flat, [-1, self.action_size, len(ACTION_VALUES)])
        self.policy = tf.nn.softmax(policy_raw, dim=2, name='policy_softmax')
        shifted_actions = self.action_input + tf.constant(1, dtype=tf.int8)
        shifted_actions = tf.cast(shifted_actions, tf.uint8)
        onehot_action_input = tf.one_hot(indices=shifted_actions, depth=len(ACTION_VALUES), on_value=1.0, off_value=0.0, axis=2, dtype=tf.float32,)
        self.q_est = tf.reduce_sum(self.policy * onehot_action_input, axis=[1,2], name='q_est')

        # tf.summary.image('q_est', self.q_est)
        # tf.summary.image('onehot_action_input', onehot_action_input)
        # policy = tf.argmax(self.policy, axis=2, name='policy_max')
        # self.output_action = policy - tf.constant(1, dtype=tf.int64)
        self.value = self._gen_layer(mixed2, in_size, 1, 'value', use_relu=False    )

    def _init_general_state_layer(self):
        in_size = self.general_state_size
        out_size = round(in_size * GENERAL_STATE_LAYER_1_FACTOR)
        self.general_state_hidden1 = self._gen_layer(self.general_state_input, in_size, out_size, 'general_state')

    # def _init_action_layers(self):
    #     in_size = self.action_size
    #     out_size = round(self.action_size * ACTION_LAYER_1_FACTOR)
    #     self.action_hidden1 = self._gen_layer(self.action_input, in_size, out_size, 'action1')
    #     # in_size = out_size
    #     # out_size = round(out_size * ACTION_LAYER_2_FACTOR)
    #     # self.action_hidden2 = self._gen_layer(action_hidden1, in_size, out_size, 'action2')

    def _init_legs_layers(self):
        in_size = self.leg_state_size
        out_size = round(self.leg_state_size * STATE_LAYER_1_FACTOR)
        weight_inititals = tf.truncated_normal(shape=[in_size, out_size], stddev=0.1)
        weights1 = tf.Variable(weight_inititals, name='shared_weights_1')
        self._variable_summaries(weights1, 'shared_weights_1')
        self.weights.append(weights1)
        bias_inititals = tf.truncated_normal(shape=[out_size], stddev=0.1)
        biases1 = tf.Variable(bias_inititals, name='shared_biases_1')
        self._variable_summaries(biases1, 'shared_biases_1')

        in_size = out_size
        out_size = round(out_size * STATE_LAYER_2_FACTOR)
        weight_inititals = tf.truncated_normal(shape=[in_size, out_size], stddev=0.1)
        weights2 = tf.Variable(weight_inititals, name='shared_weights_2')
        self._variable_summaries(weights2, 'shared_weights_2')
        self.weights.append(weights2)

        bias_inititals = tf.truncated_normal(shape=[out_size], stddev=0.1)
        biases2 = tf.Variable(bias_inititals, name='shared_biases_2')
        self._variable_summaries(biases2, 'shared_biases_2')

        self.leg_hiddens2 = []
        for i in range(self.num_of_legs):
            with tf.name_scope('layer_leg_%d_1' % i):
                leg_hidden1 = self._gen_layer_from_vars(self.legs_states_input[i], weights1, biases1, 'leg_%d_1' % i)
            with tf.name_scope('layer_leg_%d_2' % i):
                leg_hidden2 = self._gen_layer_from_vars(leg_hidden1, weights2, biases2, 'leg_%d_2' % i)
                self.leg_hiddens2.append(leg_hidden2)

    def _variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def _gen_layer_from_vars(self, input_layer, weights, biases, layer_name, use_relu=True):
        with tf.name_scope(layer_name):
            print('Layer: %s (%d -> %d)' % (layer_name, weights.get_shape()[0], weights.get_shape()[1]))
            out = tf.matmul(input_layer, weights) + biases
            tf.summary.histogram('pre_activations', out)
            if use_relu:
                out = tf.nn.relu(out)
                tf.summary.histogram('post_activations', out)
        return out

    def _gen_layer(self, input_layer, in_size, out_size, layer_name, use_relu=True):
        with tf.name_scope(layer_name):
            # xavier_init = tf.contrib.layers.xavier_initializer()
            # weights = tf.Variable(xavier_init([in_size, out_size]), name='weights')
            # biases = tf.Variable(xavier_init([out_size]), name='biases')

            weight_inititals = tf.truncated_normal(shape=[in_size, out_size], stddev=0.1)
            weights = tf.Variable(weight_inititals, name='weights')
            self._variable_summaries(weights, 'weights')
            bias_inititals = tf.truncated_normal(shape=[out_size], stddev=0.1)
            biases = tf.Variable(bias_inititals, name='biases')
            self._variable_summaries(biases, 'biases')
            self.weights.append(weights)
        return self._gen_layer_from_vars(input_layer, weights, biases, layer_name, use_relu)

