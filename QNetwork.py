import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random, math, itertools

class QNetwork(object):

    def __init__(self, state_sizes, action_size):
        self.total_q = 0
        STATE_LAYER_1_FACTOR = 2
        STATE_LAYER_2_FACTOR = 2
        LEARNING_RATE = 0.01
        LA1 = 20
        LA2 = 10
        MIX1 = 15
        BETA_W = 10 ** -3
        self.weights = []
        self.action_size = action_size
        self.state_sizes = state_sizes
        with tf.name_scope('input_state'):
            self.state_inputs = [tf.placeholder(shape=[None, part_size], dtype=tf.float32, name="input_state_part") for part_size in state_sizes]

        self.action_input = tf.placeholder(shape=[None, action_size],dtype=tf.float32, name="input_action")
        state_layer1_sizes = [part_size * STATE_LAYER_1_FACTOR for part_size in state_sizes]
        state_layer2_sizes = [part_size * STATE_LAYER_2_FACTOR for part_size in state_sizes]
        with tf.name_scope('state_layer1'):
            state_hiddens1 = [self.gen_layer(self.state_inputs[i], state_sizes[i], state_layer1_sizes[i], 'state1_%d' % i) for i in range(len(state_sizes))]
        with tf.name_scope('state_layer2'):
            state_hiddens2 = [self.gen_layer(state_hiddens1[i], state_layer1_sizes[i], state_layer2_sizes[i], 'state2_%d' % i) for i in range(len(state_sizes))]
        action_hidden1 = self.gen_layer(self.action_input, action_size, LA1, 'action1')
        action_hidden2 = self.gen_layer(action_hidden1, LA1, LA2, 'action2')
        with tf.name_scope('affine1'):
            merged = tf.concat(state_hiddens2 + [action_hidden2], axis=1, name="merged")
            self.variable_summaries(merged)
            mixed = self.gen_layer(merged, LA2 + sum(state_layer2_sizes), MIX1, 'affine1')
        self.Q_est = self.gen_layer(mixed, MIX1, 1, 'q_out', False)

        #Then combine them together to get our final Q-values.
        # self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32, name='target_q')
        with tf.name_scope('error'):
            td_error = tf.square(self.targetQ - self.Q_est)
            with tf.name_scope('regularization'):
                self.regularizers = sum([tf.nn.l2_loss(weight) for weight in self.weights])

        # tf.summary.scalar('error', td_error)
        # tf.summary.scalar('regularizer', self.regularizers)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(td_error + BETA_W * self.regularizers)
        tf.summary.scalar('loss', self.loss)
        trainer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
        self.update_model = trainer.minimize(self.loss)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def gen_layer(self, input_layer, in_size, out_size, layer_name, use_relu = True ):
        with tf.name_scope(layer_name):
            # xavier_init = tf.contrib.layers.xavier_initializer()
            # weights = tf.Variable(xavier_init([in_size, out_size]), name='weights')
            # biases = tf.Variable(xavier_init([out_size]), name='biases')

            weight_inititals = tf.truncated_normal(shape=[in_size, out_size], stddev=0.1)
            weights = tf.Variable(weight_inititals, name='weights')
            self.variable_summaries(weights)
            bias_inititals = tf.truncated_normal(shape=[out_size], stddev=0.1)
            biases = tf.Variable(bias_inititals, name='biases')
            self.variable_summaries(biases)
            self.weights.append(weights)
            out = tf.matmul(input_layer, weights) + biases
            tf.summary.histogram('pre_activations', out)
            if use_relu:
                out = tf.nn.relu(out)
            return out

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
        # import pdb;pdb.set_trace()
        feed_dict = self.states_to_feed_dict(states, actions)
        return sess.run(self.Q_est, feed_dict=feed_dict)

    def states_to_feed_dict(self, states, actions):
        split_states = np.split(states, np.cumsum(self.state_sizes), 1)[:-1]
        feed_dict = dict(zip(self.state_inputs, split_states))
        feed_dict[self.action_input] = actions
        return feed_dict