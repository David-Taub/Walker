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
        LA2 = 20
        MIX1 = 25
        BETA_W = 10 ** -5
        self.weights = []
        self.action_size = action_size
        self.state_sizes = state_sizes
        self.state_inputs = [tf.placeholder(shape=[None, part_size],dtype=tf.float32) for part_size in state_sizes]

        self.action_input = tf.placeholder(shape=[None, action_size],dtype=tf.float32)
        state_layer1_sizes = [part_size * STATE_LAYER_1_FACTOR for part_size in state_sizes]
        state_layer2_sizes = [part_size * STATE_LAYER_2_FACTOR for part_size in state_sizes]
        state_hiddens1 = [self.gen_layer(self.state_inputs[i], state_sizes[i], state_layer1_sizes[i]) for i in range(len(state_sizes))]
        state_hiddens2 = [self.gen_layer(state_hiddens1[i], state_layer1_sizes[i], state_layer2_sizes[i]) for i in range(len(state_sizes))]
        action_hidden1 = self.gen_layer(self.action_input, action_size, LA1)
        action_hidden2 = self.gen_layer(action_hidden1, LA1, LA2)
        merged = tf.concat(state_hiddens2 + [action_hidden2], axis=1)
        mixed = self.gen_layer(merged, LA2 + sum(state_layer2_sizes), MIX1)
        self.Q_est = self.gen_layer(mixed, MIX1, 1, False)

        #Then combine them together to get our final Q-values.
        # self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        td_error = tf.square(self.targetQ - self.Q_est)
        self.regularizers = sum([tf.nn.l2_loss(weight) for weight in self.weights])
        self.action_regularizer = tf.nn.l2_loss(self.action_input)
        self.loss = tf.reduce_mean(td_error + BETA_W * self.regularizers)
        trainer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
        self.update_model = trainer.minimize(self.loss)

    def gen_layer(self, input_layer, in_size, out_size, use_relu = True):
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.weights.append(tf.Variable(xavier_init([in_size,out_size]), name='weights'))
        biases = tf.Variable(xavier_init([out_size]), name='biases')
        out = tf.matmul(input_layer, self.weights[-1]) + biases
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
        feed_dict = self.states_to_feed_dict(states)
        feed_dict[self.action_input] = actions
        return sess.run(self.Q_est, feed_dict=feed_dict)

    def states_to_feed_dict(self, states):
        split_states = np.split(states, np.cumsum(self.state_sizes), 1)[:-1]
        feed_dict = dict(zip(self.state_inputs, split_states))
        return feed_dict