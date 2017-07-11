
import random, math, sys, pickle, os
import numpy as np
from Panda3dApp import Panda3dApp
from Shape import Shape, Spider

TIME_STEP = 0.1
PHYSICAL_STEPS_IN_STEP = 5
class Walker(object):

    def __init__(self, is_displaying):
        self.last_score = 0
        self.app = Panda3dApp(self, is_displaying)
        self._init_shape()


    def _get_pickle_path(self):
        return os.path.join(os.path.dirname(sys.argv[0]), 'shape.pickle')

    def reset(self):
        self.app.restart_bones_position()
        self.last_score = self.score()
        return self.get_state()

    def _init_shape(self):
        pickle_path = self._get_pickle_path()
        if os.path.isfile(pickle_path):
            with open(self._get_pickle_path(), 'rb') as f:
                self.shape = pickle.load(f)
            self.app.load_shape(self.shape)
            return
        print("Generating shape")
        self.shape = Spider()
        # self.shape = WormShape()
        self.app.load_shape(self.shape)
        print("Positioning shape in start posture")
        com = self.app.get_com()
        while True:
            self.step([0] * len(self.shape.joints), False)
            new_com = self.app.get_com()
            if (com - new_com).length() == 0:
                break
            com = new_com

        self.app.save_shape_posture()
        self.save_shape()
        self.app.restart_bones_position()

    def action_size(self):
        return len(self.shape.joints)

    def save_shape(self):
        print("Saving shape to %s" % self._get_pickle_path())
        try:
            with open(self._get_pickle_path(), 'wb') as f:
                pickle.dump(self.shape, f)
                return
        except IOError:
            print("DUMP FAILED! %s" % self._get_pickle_path())

    def step(self, action, add_debug = True):
        self.app.step(action, TIME_STEP, PHYSICAL_STEPS_IN_STEP)
        state, reward = self.get_state(), self.get_reward()
        if add_debug:
            self.app.debug_screen_print(action, state, reward, self.score())
        return state, reward

    def get_state_sizes(self):
        return self.get_state(True)

    def get_state(self, return_sizes = False):
        state = np.array([])
        if return_sizes:
            sizes = []
        joints_angles = np.array(self.app.get_joint_angles()) / 180
        bones_z = np.array(self.app.get_bones_z())
        bones_contacts = self.app.get_contacts()

        for part in self.shape.parts:
            # todo: make each bone position relative to it part com
            state_part = np.array(self.app.get_bones_positions())[part, 0] - self.app.get_com()
            state_part = np.append(state_part, bones_contacts[part])
            state_part = np.append(state_part, joints_angles[part[1:]-1])
            if return_sizes:
                sizes.append(state_part.shape[0])
            state = np.append(state, state_part)

        if return_sizes:
            return sizes
        return np.array(state)

    def score(self):
        # return self.app.get_com().length()
        return self.app.get_com()[2] + self.app.get_com()[1]

    def get_reward(self):
        new_score = self.score()
        ret = new_score - self.last_score
        self.last_score = new_score
        return ret


