# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import itertools

STATE_LAYER_1_FACTOR = 2
STATE_LAYER_2_FACTOR = 1
GENERAL_STATE_LAYER_1_FACTOR = 1
ACTION_LAYER_1_FACTOR = 2
ACTION_LAYER_2_FACTOR = 1
MERGED_LAYER_FATOR_1 = 0.7
MERGED_LAYER_FATOR_2 = 0.5


class QNetwork(object):

    def __init__(self, state_sizes, action_size):
        print('Building Q network')
        self.total_q = 0
        self.LEARNING_RATE = 0.001
        self.BETA_W = 10 ** -1
        self.weights = []
        self.action_size = action_size
        self.general_state_size = state_sizes[0]
        self.num_of_legs = state_sizes[1]
        self.leg_state_size = state_sizes[2]

        self._init_inputs()
        self._init_legs_layers()
        self._init_action_layers()
        self._init_general_state_layer()
        self._init_merge_and_q()

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32, name='target_q')
        with tf.name_scope('error'):
            td_error = tf.square(self.targetQ - self.Q_est)
            with tf.name_scope('regularization'):
                self.regularizers = tf.abs(sum([tf.nn.l2_loss(weight) for weight in self.weights]) - 3)

        tf.summary.scalar('regularizer', self.regularizers)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(td_error + self.BETA_W * self.regularizers)
        tf.summary.scalar('loss', self.loss)
        trainer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.update_model = trainer.minimize(self.loss)

    def _init_inputs(self):
        with tf.name_scope('input_action'):
            self.action_input = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name="input_action")
        with tf.name_scope('input_state'):
            shape = [None, self.general_state_size]
            self.general_state_input = tf.placeholder(shape=shape, dtype=tf.float32, name="input_general_state")
            self.legs_states_input = []
            shape = [None, self.leg_state_size]
            for i in range(self.num_of_legs):
                self.legs_states_input.append(tf.placeholder(shape=shape, dtype=tf.float32, name="input_general_state"))

    def _init_merge_and_q(self):
        with tf.name_scope('merged_1'):
            to_merge = [self.action_hidden1] + self.leg_hiddens2 + [self.general_state_hidden1]
            merged = tf.concat(to_merge, axis=1, name="merged")
            self.variable_summaries(merged, 'merged')
            in_size = merged.get_shape()[1].value
            out_size = round(in_size * MERGED_LAYER_FATOR_1)
            mixed1 = self.gen_layer(merged, in_size, out_size, 'affine1')
            in_size = out_size
            out_size = round(in_size * MERGED_LAYER_FATOR_2)
            mixed2 = self.gen_layer(mixed1, in_size, out_size, 'affine2')
        in_size = out_size
        self.Q_est = self.gen_layer(mixed2, in_size, 1, 'q_out', False)

    def _init_general_state_layer(self):
        in_size = self.general_state_size
        out_size = round(in_size * GENERAL_STATE_LAYER_1_FACTOR)
        self.general_state_hidden1 = self.gen_layer(self.general_state_input, in_size, out_size, 'general_state')

    def _init_action_layers(self):
        in_size = self.action_size
        out_size = round(self.action_size * ACTION_LAYER_1_FACTOR)
        self.action_hidden1 = self.gen_layer(self.action_input, in_size, out_size, 'action1')
        # in_size = out_size
        # out_size = round(out_size * ACTION_LAYER_2_FACTOR)
        # self.action_hidden2 = self.gen_layer(action_hidden1, in_size, out_size, 'action2')

    def _init_legs_layers(self):
        in_size = self.leg_state_size
        out_size = round(self.leg_state_size * STATE_LAYER_1_FACTOR)
        weight_inititals = tf.truncated_normal(shape=[in_size, out_size], stddev=0.1)
        weights1 = tf.Variable(weight_inititals, name='shared_weights_1')
        self.variable_summaries(weights1, 'shared_weights_1')
        self.weights.append(weights1)
        bias_inititals = tf.truncated_normal(shape=[out_size], stddev=0.1)
        biases1 = tf.Variable(bias_inititals, name='shared_biases_1')
        self.variable_summaries(biases1, 'shared_biases_1')

        in_size = out_size
        out_size = round(out_size * STATE_LAYER_2_FACTOR)
        weight_inititals = tf.truncated_normal(shape=[in_size, out_size], stddev=0.1)
        weights2 = tf.Variable(weight_inititals, name='shared_weights_2')
        self.variable_summaries(weights2, 'shared_weights_2')
        self.weights.append(weights2)

        bias_inititals = tf.truncated_normal(shape=[out_size], stddev=0.1)
        biases2 = tf.Variable(bias_inititals, name='shared_biases_2')
        self.variable_summaries(biases2, 'shared_biases_2')

        self.leg_hiddens2 = []
        for i in range(self.num_of_legs):
            with tf.name_scope('layer_leg_%d_1' % i):
                leg_hidden1 = self.gen_layer_from_vars(self.legs_states_input[i], weights1, biases1, 'leg_%d_1' % i)
            with tf.name_scope('layer_leg_%d_2' % i):
                leg_hidden2 = self.gen_layer_from_vars(leg_hidden1, weights2, biases2, 'leg_%d_2' % i)
                self.leg_hiddens2.append(leg_hidden2)

    def variable_summaries(self, var, name):
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

    def gen_layer_from_vars(self, input_layer, weights, biases, layer_name, use_relu=True):
        with tf.name_scope(layer_name):
            print('Layer: %s (%d -> %d)' % (layer_name, weights.get_shape()[0], weights.get_shape()[1]))
            out = tf.matmul(input_layer, weights) + biases
            tf.summary.histogram('pre_activations', out)
            if use_relu:
                out = tf.nn.relu(out)
                tf.summary.histogram('post_activations', out)
        return out

    def gen_layer(self, input_layer, in_size, out_size, layer_name, use_relu=True):
        with tf.name_scope(layer_name):
            # xavier_init = tf.contrib.layers.xavier_initializer()
            # weights = tf.Variable(xavier_init([in_size, out_size]), name='weights')
            # biases = tf.Variable(xavier_init([out_size]), name='biases')

            weight_inititals = tf.truncated_normal(shape=[in_size, out_size], stddev=0.1)
            weights = tf.Variable(weight_inititals, name='weights')
            self.variable_summaries(weights, 'Weights')
            bias_inititals = tf.truncated_normal(shape=[out_size], stddev=0.1)
            biases = tf.Variable(bias_inititals, name='biases')
            self.variable_summaries(biases, 'biases')
            self.weights.append(weights)
        return self.gen_layer_from_vars(input_layer, weights, biases, layer_name, use_relu)

    #get state, predict action by simple per-dimension ascent
    # states - batch_size x state_size
    # actions - batch_size x state_size
    def predict(self, states, sess):
        REPETITIONS = 1
        action_values = np.array([0, -1, 1])
        batch_size = states.shape[0]
        actions = np.zeros((batch_size, self.action_size))
        action_indices = list(range(self.action_size))
        random.shuffle(action_indices)
        pool = itertools.cycle(action_indices)
        for i in range(self.action_size * REPETITIONS):
            index = next(pool)
            q_options = [self.get_q_action_modified(states, actions, index, new_val, sess) for new_val in action_values]
            q_options = np.concatenate(q_options, axis=1)
            actions[:, index] = action_values[np.argmax(q_options, axis=1)]
        if batch_size == 1:
            self.total_q += np.max(q_options, axis=1)[0]
        return actions

    def pop_total_q(self):
        ret = self.total_q
        self.total_q = 0
        return ret

    def get_q_action_modified(self, states, actions, index, new_val, sess):
        modified_actions = np.copy(actions)
        modified_actions[:, index] = new_val
        return self.calc_q(states, modified_actions, sess)

    def calc_q(self, states, actions, sess):
        feed_dict = self.states_to_feed_dict(states, actions)
        return sess.run(self.Q_est, feed_dict=feed_dict)

    def states_to_feed_dict(self, states, actions):
        # import pdb; pdb.set_trace()
        # split_states = np.split(states, np.cumsum(self.state_sizes), 1)[:-1]
        # feed_dict = dict(zip(self.state_inputs, split_states))
        feed_dict = {}
        feed_dict[self.general_state_input] = states[:, :self.general_state_size]
        for i in range(self.num_of_legs):
            start = self.general_state_size + i * self.leg_state_size
            end = self.general_state_size + (i + 1) * self.leg_state_size
            feed_dict[self.legs_states_input[i]] = states[:, start:end]
        feed_dict[self.action_input] = actions
        return feed_dict
