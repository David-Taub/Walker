# python C:\Users\booga\Dropbox\bio\projects\Walker\reinfLearn.py

import numpy as np
import os, random, math, itertools, sys
import tensorflow as tf

import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from Walker import Walker



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size          = 32 #How many experiences to use for each training step.
update_freq         = 4 #How often to perform a training step.
y                   = .99 #Discount factor on the target Q-values
START_EXPLOIT_PROB  = 1 #Starting chance of random action
END_EXPLOIT_PROB    = 0.1 #Final chance of random action
NUM_EPISODES        = 20000 #How many episodes of game environment to train network with.
ANNEALING_STEPS     = 10000 #How many steps of training to reduce START_EXPLOIT_PROB to END_EXPLOIT_PROB.
PRE_TRAIN_STEPS     = 1000 #How many steps of random actions before training begins.
EPISODE_LENGTH      = 200 #The max allowed length of our episode.
BASE_DIR            = os.path.dirname(sys.argv[0])+"/dqn" #The path to save our model to.
h_size              = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau                 = 0.001 #Rate to update target network toward primary network
SHOW_WHILE_TRAINING = False
class QNetwork(object):
    def gen_layer(self, input_layer, in_size, out_size):
        weights = tf.Variable(
            tf.truncated_normal([in_size, out_size],
                                stddev=1.0 / math.sqrt(float(in_size))),
            name='weights1')
        biases = tf.Variable(tf.zeros([out_size]),
                             name='biases')
        return tf.nn.relu(tf.matmul(input_layer, weights) + biases)

    def __init__(self, state_size, action_size):
        LS1 = 10
        LA1 = 10
        MIX1 = 20
        final = 10
        self.action_size = action_size
        self.state_size = state_size
        self.state_input = tf.placeholder(shape=[None, state_size],dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, action_size],dtype=tf.float32)

        state_hidden1 = self.gen_layer(self.state_input, state_size, LS1)
        # state_hidden2 = self.gen_layer(state_hidden1, state_size, LS2)
        action_hidden1 = self.gen_layer(self.action_input, action_size, LA1)
        merged = tf.concat([state_hidden1, action_hidden1], axis=1)
        mixed = self.gen_layer(merged, LA1 + LS1, MIX1)
        self.Q_est = self.gen_layer(mixed, MIX1, 1)

        #Then combine them together to get our final Q-values.
        # self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.td_error = tf.square(self.targetQ - self.Q_est)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_model = self.trainer.minimize(self.loss)
    
    #get state, predict action by simple per-dimension ascent
    # states - batch_size x state_size
    # actions - batch_size x state_size
    def predict(self, states, sess):
        action_values = np.array([-1, 0, 1])
        batch_size = states.shape[0]
        actions = np.zeros((batch_size, self.action_size))
        action_indices = list(range(self.action_size))
        random.shuffle(action_indices)
        pool = itertools.cycle(action_indices)
        for i in range(self.action_size):
            index = next(pool)
            q_options = [self.get_q_action_modified(states, actions, index, new_val, sess) for new_val in action_values]
            q_options = np.concatenate(q_options, axis=1)
            actions[:, index] = action_values[np.argmax(q_options, axis=1)]
        return actions
        
    def get_q_action_modified(self, states, actions, index, new_val, sess):
        modified_actions = np.copy(actions)
        modified_actions[:, index] = new_val
        return self.calc_q(states, modified_actions, sess)

    def calc_q(self, states, actions, sess):
        return sess.run(self.Q_est, feed_dict={self.state_input:states, self.action_input:actions})




class ExperienceBuffer(object):
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size,4])

class Learner(object):
    def gen_target_ops(self):
        tfVars = tf.trainable_variables()
        total_vars = len(tfVars)
        self.target_ops = []
        for idx,var in enumerate(tfVars[0:total_vars//2]):
            new_val = (var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())
            self.target_ops.append(tfVars[idx+total_vars//2].assign(new_val))
        

    def update_target(self):
        #Set the target network to be equal to the primary network.
        for op in self.target_ops:
            self.sess.run(op)

    def _load_model(self):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(BASE_DIR)
        print('Loading %s' % ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def __init__(self):
        tf.reset_default_graph()
        self.walker = Walker()
        self.exploit_prob = START_EXPLOIT_PROB
        self.mainQN = QNetwork(self.walker.state_size(), self.walker.action_size())
        self.targetQN = QNetwork(self.walker.state_size(), self.walker.action_size())
        self.saver = tf.train.Saver()
        self.gen_target_ops()
        self.training_buffer = ExperienceBuffer()
        #Set the rate of random action decrease. 
        self.step_drop = (START_EXPLOIT_PROB - END_EXPLOIT_PROB)/ANNEALING_STEPS
        #create lists to contain total rewards and steps per episode
        self.episodes_rewards_list = []
        self.total_steps = 0
        self.sess = tf.Session()

        #Make a path for our model to be saved in.
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
            self.load_model = False
        else:
            self.load_model = True
        print("Start of training")
        self.sess.run(tf.global_variables_initializer())
        if self.load_model:
            self._load_model()


    def train(self):
        self.update_target() 
        for i in range(NUM_EPISODES):
            print("Episode %d" % i)
            self._run_episode()
            #Periodically save the model. 
            if i % 50 == 0:
                self.saver.save(self.sess, BASE_DIR+'/model-'+str(i)+'.cptk')
                print("Saved Model")
            if len(self.episodes_rewards_list) % 10 == 0:
                print(i, self.total_steps, np.mean(self.episodes_rewards_list[-10:]), self.exploit_prob)
        self.saver.save(self.sess, BASE_DIR+'/model-'+str(i)+'.cptk')

    def _run_step(self, state, replay_buffer):
        #Choose an action by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) < self.exploit_prob or self.total_steps < PRE_TRAIN_STEPS:
            action = np.random.randint(-1, 2, (self.mainQN.action_size))
        else:
            action = self.mainQN.predict(np.vstack(state).transpose(), self.sess)[0]
        next_state, reward = self.walker.step(action)
        
        replay_buffer.add(np.reshape(np.array([state, action, reward, next_state]), [1, 4])) #Save the experience to our episode buffer.
        return next_state, reward

    def _batch_train_QN(self):
        train_batch = self.training_buffer.sample(batch_size) #Get a random batch of experiences.
        #Below we perform the Double-DQN update to the target Q-values
        actions = self.mainQN.predict(np.vstack(train_batch[:, 3]), self.sess)

        doubleQ = self.sess.run(self.targetQN.Q_est, feed_dict={self.targetQN.state_input: np.vstack(train_batch[:, 3]),
                                                                self.targetQN.action_input: actions})[:,0]
        targetQ = train_batch[:, 2] + (y * doubleQ)
        # import pdb; pdb.set_trace()
        #Update the network with our target values.
        self.sess.run(self.mainQN.update_model, \
            feed_dict={self.mainQN.state_input: np.vstack(train_batch[:,0]), 
                       self.mainQN.action_input: np.vstack(train_batch[:,1]),
                       self.mainQN.targetQ: targetQ})
        self.update_target() #Set the target network to be equal to the primary network.

    def _run_episode(self):
        episode_buffer = ExperienceBuffer()
        #Reset environment and get first new observation
        state = self.walker.reset(SHOW_WHILE_TRAINING)
        episode_rewards = 0
        for j in range(EPISODE_LENGTH):
            state, reward  = self._run_step(state, episode_buffer)
            self.total_steps += 1
            episode_rewards += reward
            if self.total_steps <= PRE_TRAIN_STEPS:
                continue
            if self.exploit_prob > END_EXPLOIT_PROB:
                self.exploit_prob -= self.step_drop
            if self.total_steps % (update_freq) == 0:
                self._batch_train_QN()
        self.training_buffer.add(episode_buffer.buffer)
        self.episodes_rewards_list.append(episode_rewards)

    def show(self):
        state = self.walker.reset(True)
        for i in range(EPISODE_LENGTH):
            action = self.mainQN.predict(np.vstack(state).transpose(), self.sess)[0]
            state, _ = self.walker.step(action)

def main():
  l = Learner()
  l.train()
  l.show()
  l.show()


if __name__ == "__main__":
  main()