
import random, math, sys, pickle, os
import numpy as np
import MyApp
from Shape import Shape
class Walker(object):
    BUFFER_LENGTH = 0.6
    TIME_STEP = 0.1
    PHYSICAL_STEPS_IN_STEP = 10
    NUM_OF_BONES = 8

    def __init__(self, is_displaying):
        self.last_score = None
        self.app = MyApp.MyApp(self, is_displaying)
        self.load_world()


    def _get_pickle_path(self):
        return os.path.join(os.path.dirname(sys.argv[0]), 'shape.pickle')

    def reset(self):
        self.app.remove_shape()
        self.last_score = None
        self.load_world()
        return self.get_state()

    def load_world(self):
        self.app.init_plane()
        pickle_path = self._get_pickle_path()
        if os.path.isfile(pickle_path):
            with open(self._get_pickle_path(), 'rb') as f:
                self.shape = pickle.load(f)
            self._build_bones_and_joints()
        else:
            self.shape = Shape(self.NUM_OF_BONES)
            self._build_bones_and_joints()
            for i in range(300):
                self.step([0] * len(self.joints))
            self.shape.start_score = self.score()
            self.save_shape()


    def _build_bones_and_joints(self):
        self.joints = []
        self.bones = []
        for i in range(self.shape.N):
            bone = Bone(self.app, self.shape.lengths[i], self.shape.positions[i], self.shape.widths[i], self.shape.heights[i], i)
            self.bones.append(bone)
            if self.shape.connections[i] != -1:
                joint = Joint(self.bones[self.shape.connections[i]], bone, self.app, self.shape.headings[i], self.shape.pitches[i])
                self.joints.append(joint)


    def state_size(self):
        return len(self.get_state())

    def action_size(self):
        return len(self.joints)

    def save_shape(self):
        self.shape.positions = self.app.get_bones_positions()
        try:
            with open(self._get_pickle_path(), 'wb') as f:
                return pickle.dump(self.shape, f)
        except IOError:
            print("DUMP FAILED! %s" % self._get_pickle_path())

    def step(self, action):
        for i in range(self.PHYSICAL_STEPS_IN_STEP):
            self.app.physical_step(action, self.TIME_STEP)
        state, reward, score = self.get_state(), self.get_reward(), self.score()
        self.app.debug_screen_print(action, state, reward, score)
        return state, reward

    def get_state(self):
        state = []
        state += self.app.get_joint_angles()
        state += self.app.get_bones_z()
        state += self.app.head_hpr()
        return np.array(state)

    def gen_actions(self):
        if self.physical_steps_done_in_step % self.PHYSICAL_STEPS_IN_STEP == 0:
            self.physical_steps_done_in_step = 0
            self.action_in_step = self.action_generator(self.get_state())
        self.physical_steps_done_in_step += 1
        return self.action_in_step

    def score(self):
        return self.app.get_com()[1] - self.shape.start_score
        # return self.app.get_com()[2] + self.app.get_com()[1]

    def get_reward(self):
        new_score = self.score()
        ret = new_score - self.last_score if self.last_score is not None else 0
        self.last_score = new_score
        return ret

class Joint(object):
    def __init__(self, parent_bone, child_bone, app, heading, pitch):
        hpr = [heading, pitch, 0]
        hpr = [0, pitch, 0]
        pos = 1
        self.constraint = app.add_joint(parent_bone, child_bone, hpr, pos)
        self.parent_bone = parent_bone
        self.child_bone = child_bone


class Bone(object):
    def __init__(self, app, length, position, width, height, index):
        self.has_joint_ball = False
        self.width = width
        self.height = height
        self.length = length
        app.init_bone(self, position, index)

