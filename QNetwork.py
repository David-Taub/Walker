import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random, math, itertools

class QNetwork(object):
    def gen_layer(self, input_layer, in_size, out_size, use_relu = True):
        self.weights.append(tf.Variable(
            tf.truncated_normal([in_size, out_size],
                                stddev=1.0 / math.sqrt(float(in_size))),
            name='weights'))
        biases = tf.Variable(tf.zeros([out_size]),
                             name='biases')
        out = tf.matmul(input_layer, self.weights[-1]) + biases
        if use_relu:
            out = tf.nn.relu(out)
        return out

    def __init__(self, state_size, action_size):
        LS1 = 40
        LS2 = 30
        LA1 = 20
        MIX1 = 20
        BETA_W = 0.5
        self.weights = []
        self.action_size = action_size
        self.state_size = state_size
        self.state_input = tf.placeholder(shape=[None, state_size],dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, action_size],dtype=tf.float32)

        state_hidden1 = self.gen_layer(self.state_input, state_size, LS1)
        state_hidden2 = self.gen_layer(state_hidden1, LS1, LS2)
        action_hidden1 = self.gen_layer(self.action_input, action_size, LA1)
        merged = tf.concat([state_hidden2, action_hidden1], axis=1)
        mixed = self.gen_layer(merged, LA1 + LS2, MIX1)
        self.Q_est = self.gen_layer(mixed, MIX1, 1, False)

        #Then combine them together to get our final Q-values.
        # self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        td_error = tf.square(self.targetQ - self.Q_est)
        self.regularizers = sum([tf.nn.l2_loss(weight) for weight in self.weights])
        self.action_regularizer = tf.nn.l2_loss(self.action_input)
        self.loss = tf.reduce_mean(td_error + BETA_W * self.regularizers)
        trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_model = trainer.minimize(self.loss)

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
        qs = []
        for i in range(self.action_size * REPETITIONS):
            index = next(pool)
            q_options = [self.get_q_action_modified(states, actions, index, new_val, sess) for new_val in action_values]
            q_options = np.concatenate(q_options, axis=1)
            actions[:, index] = action_values[np.argmax(q_options, axis=1)]
            qs.append(np.max(q_options, axis=1)[0])
        print("%.2f" % (qs[-1]))
        return actions

    def get_q_action_modified(self, states, actions, index, new_val, sess):
        modified_actions = np.copy(actions)
        modified_actions[:, index] = new_val
        return self.calc_q(states, modified_actions, sess)

    def calc_q(self, states, actions, sess):
        return sess.run(self.Q_est, feed_dict={self.state_input:states, self.action_input:actions})
