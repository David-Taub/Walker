
import random, math, sys, pickle, os
import numpy as np
from Panda3dApp import Panda3dApp
from Shape import Shape, Spider

PHISICAL_STEP_TIME = 0.1
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
        # self.shape = Shape()
        # self.shape = WormShape()
        self.app.load_shape(self.shape)
        print("Positioning shape in start posture")
        coms = [self.app.get_com()]
        while True:
            self.step([0] * len(self.shape.joints), False)
            coms.append(self.app.get_com())
            if len(coms) > 10 and (coms[-1] - coms[-2]).length() == 0:
                break

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

    def step(self, action, add_debug=True):
        self.app.step(action, PHISICAL_STEP_TIME, PHYSICAL_STEPS_IN_STEP)
        state, reward = self.get_state(), self.get_reward()
        if add_debug:
            self.app.debug_screen_print(action, state, reward, self.score())
        return state, reward

    def get_state_sizes(self):
        return self.get_state(True)

    # state - <body> <legs>
    # body - ground contacts, body joint angles, body com, leg com, leg com...
    # <legs> = <leg> [ leg, leg, leg...]
    # <leg> = ground contacts, joint angles
    def get_state(self, get_sizes=False):
        joints_angles = [angle / 180 for angle in self.app.get_joint_angles()]
        bones_contacts = self.app.get_contacts()
        bones_pos = self.app.get_bones_positions()
        body_com = self.app.get_com(self.shape.body)
        legs_coms = [self.app.get_com(leg) - body_com for leg in self.shape.legs]

        general_state = []
        general_state += [joints_angles[i] for i in self.shape.body[:-1]]    # body joints angle
        general_state += [bones_contacts[i] for i in self.shape.body]        # body bones ground contact
        general_state += [bones_pos[i][2] for i in self.shape.body]          # body bones z
        general_state += [bones_pos[i][1] - body_com[1] for i in self.shape.body]  # body bones y offset
        for leg_com in legs_coms:                                            # legs com
            general_state += list(leg_com)
        for leg in self.shape.legs:
            hip_angle = joints_angles[leg[0]]
            general_state.append(hip_angle)

        leg_states = []
        for leg in self.shape.legs:
            leg_state = []
            leg_state += [bones_contacts[i] for i in leg]
            leg_state += [joints_angles[i] for i in leg[1:-1]]
            leg_states.append(leg_state)
        if get_sizes:
            return [len(general_state), len(self.shape.legs), len(leg_states[0])]
        return general_state + sum(leg_states, [])

    def score(self):
        # return self.app.get_com().length()
        return self.app.get_com()[0]

    def get_reward(self):
        new_score = self.score()
        ret = new_score - self.last_score
        self.last_score = new_score
        return ret


