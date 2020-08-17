import logging
import sys
import pickle
import os

# import numpy as np
from Panda3dApp import Panda3dApp
import Shape

PHISICAL_STEP_TIME = 0.1
PHYSICAL_STEPS_IN_STEP = 1


class Environment:
    def __init__(self, is_displaying):
        self.last_score = 0
        self.app = Panda3dApp(self, is_displaying)
        self._init_shape()

    # def _get_pickle_path(self):
    #     return os.path.join(os.path.dirname(sys.argv[0]), 'shape.pickle')

    def reset(self):
        self.app.restart_bones_position()
        self.last_score = self.score()
        return self.get_state()

    def _init_shape(self):
        # pickle_path = self._get_pickle_path()
        # if os.path.isfile(pickle_path):
        #     with open(self._get_pickle_path(), 'rb') as f:
        #         self.shape = pickle.load(f)
        #     self.app.load_shape(self.shape)
        #     return
        logging.debug("Generating shape")
        # self.shape = Spider()
        # self.shape = Shape()
        self.shape = Shape.Worm()
        self.app.load_shape(self.shape)
        logging.debug("Positioning shape in start posture")
        centers_of_mass = [self.app.get_center_of_mass()]
        while True:
            self.step([0] * len(self.shape.joints), False)
            centers_of_mass.append(self.app.get_center_of_mass())
            if len(centers_of_mass) > 10 and (centers_of_mass[-1] - centers_of_mass[-2]).length() == 0:
                break

        self.app.save_shape_posture()
        self.save_shape()
        self.app.restart_bones_position()

    def action_size(self):
        return len(self.shape.joints)

    def save_shape(self):
        logging.debug("Saving shape to %s" % self._get_pickle_path())
        try:
            with open(self._get_pickle_path(), 'wb') as f:
                pickle.dump(self.shape, f)
                return
        except IOError:
            logging.debug("DUMP FAILED! %s" % self._get_pickle_path())

    def step(self, action, add_debug=True):
        """
        Perform a single physical step
        """
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
        body_com = self.app.get_center_of_mass(self.shape.body)
        legs_coms = [self.app.get_center_of_mass(leg) - body_com for leg in self.shape.legs]

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
        # return self.app.get_center_of_mass().length()
        return self.app.get_center_of_mass()[0]

    def get_reward(self):
        new_score = self.score()
        ret = new_score - self.last_score
        self.last_score = new_score
        return ret
