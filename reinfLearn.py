# python C:\Users\booga\Dropbox\bio\projects\Walker\reinfLearn.py

import numpy as np
import os, random, math, itertools, sys, time
import tensorflow as tf

import matplotlib.pyplot as plt
from Walker import Walker
from QNetwork import QNetwork
from ExperienceBuffer import ExperienceBuffer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size          = 10 #How many experiences to use for each training step.
UPDATE_FREQ         = 2 #How often to perform a training step.
y                   = .99 #Discount factor on the target Q-values
START_EXPLOIT_PROB  = 1 #Starting chance of random action
END_EXPLOIT_PROB    = 0.2 #Final chance of random action
NUM_EPISODES        = 20000 #How many episodes of game environment to train network with.
ANNEALING_EPISODES  = 300 #How many steps of training to reduce START_EXPLOIT_PROB to END_EXPLOIT_PROB.
PRE_TRAIN_STEPS     = 1000 #How many steps of random actions before training begins.
EPISODE_LENGTH      = 50 #The max allowed length of our episode.
BASE_DIR            = os.path.dirname(sys.argv[0])+"/dqn" #The path to save our model to.
tau                 = 0.001 #Rate to update target network toward primary network
EPISODES_BETWEEN_SAVE = 100
EPISODES_BETWEEN_BIG_SUMMERY = 10
STOP_EPISODE_SCORE_THRESHOLD = -1

class Learner(object):
    def gen_target_ops(self):
        trainable_variables = tf.trainable_variables()
        total_vars = len(trainable_variables)
        self.target_ops = []
        for idx, var in enumerate(trainable_variables[0:total_vars // 2]):
            # convex combination of new and old values
            new_val = (var.value() * tau) + ((1 - tau) * trainable_variables[idx + total_vars // 2].value())
            self.target_ops.append(trainable_variables[idx+total_vars//2].assign(new_val))


    def update_target(self):
        #Set the target network to be equal to the primary network.
        for op in self.target_ops:
            self.sess.run(op)

    def _load_model(self):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(BASE_DIR)
        print('Loading %s' % ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def __init__(self, is_displaying, no_explore):
        self.start_time = time.time()
        tf.reset_default_graph()
        self.walker = Walker(is_displaying)
        self.explore_prob = 0 if no_explore else START_EXPLOIT_PROB
        self.mainQN = QNetwork(self.walker.state_size(), self.walker.action_size())
        self.targetQN = QNetwork(self.walker.state_size(), self.walker.action_size())
        self.saver = tf.train.Saver()
        self.gen_target_ops()
        self.training_buffer = ExperienceBuffer()
        #Set the rate of random action decrease. 
        self.episode_drop = (START_EXPLOIT_PROB - END_EXPLOIT_PROB)/ANNEALING_EPISODES      #create 60 to contain total rewards and steps per episode
        self.episodes_rewards_list = []
        self.total_steps = 0
        self.sess = tf.Session()

        #Make a path for our model to be saved in.
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
            self.load_model = False
        else:
            self.load_model = True
        self.sess.run(tf.global_variables_initializer())
        if self.load_model:
            self._load_model()

    def _episode_summery(self, index):
        elapsed = time.time() - self.start_time
        print("Episode %d - Score %.2f (%.2f seconds)" % (index, self.episodes_rewards_list[-1], elapsed))
        self.start_time = time.time()
        if index % EPISODES_BETWEEN_SAVE == 0:
            try:
                self.saver.save(self.sess, BASE_DIR+'/model-'+str(index)+'.cptk')
                print("Saved model")
            except:
                print("SAVING FAILED!")
        if index % EPISODES_BETWEEN_BIG_SUMMERY == 0:
            print("Average total reward: %.2f" % np.mean(self.episodes_rewards_list[-10:]))
            print("Total Q: %.2f" % self.mainQN.pop_total_q())
            print("Explore probability: %.2f" % self.explore_prob)
            print("L2 of weights: %.2f" % self.sess.run(self.mainQN.regularizers))

    def train(self):
        print("Start of training")
        
        # do stuff
        self.update_target() 
        for i in range(NUM_EPISODES):
            self._run_episode()
            #Periodically save the model.
            self._episode_summery(i)            
        self.saver.save(self.sess, BASE_DIR+'/model-'+str(i)+'.cptk')

    def _run_step(self, state, replay_buffer):
        #Choose an action by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) < self.explore_prob or self.total_steps < PRE_TRAIN_STEPS:
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
        state = self.walker.reset()
        episode_rewards = 0
        for j in range(EPISODE_LENGTH):
            state, reward  = self._run_step(state, episode_buffer)
            self.total_steps += 1
            episode_rewards += reward
            if self.total_steps <= PRE_TRAIN_STEPS:
                continue
            if self.total_steps % (UPDATE_FREQ) == 0:
                self._batch_train_QN()
            if self.walker.score() < STOP_EPISODE_SCORE_THRESHOLD:
                break
        if self.explore_prob > END_EXPLOIT_PROB:
            self.explore_prob -= self.episode_drop
        self.training_buffer.add(episode_buffer.buffer)
        self.episodes_rewards_list.append(episode_rewards)

    def show(self):
        state = self.walker.reset(True)
        for i in range(EPISODE_LENGTH*20):
            action = self.mainQN.predict(np.vstack(state).transpose(), self.sess)[0]
            state, _ = self.walker.step(action)

def main():
    is_displaying = len(sys.argv) > 1
    no_explore = len(sys.argv) > 2
    l = Learner(is_displaying, no_explore)
    if len(sys.argv) == 4:
        l.show()
    else:
        l.train()


if __name__ == "__main__":
  main()