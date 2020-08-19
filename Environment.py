import logging
import Shape
from numpy import linalg as LA
from Panda3dPhysics import Panda3dPhysics
from Panda3dDisplay import Panda3dDisplay

import numpy as np

MAX_STABILITY_STEPS = 200
MIN_MOVEMENT_FOR_STABILITY = 0.0001
MIN_MOVEMENT_FOR_END_EPISODE = 0.01


class Environment:
    def __init__(self):
        self.display = None
        self.physics = Panda3dPhysics()
        walker = Shape.Worm()
        self.bone_count = len(walker.bones)
        self.joints_count = len(walker.joints)
        self.physics.add_walker(walker)
        self._wait_for_stability()
        self.init_state = self.get_current_state()

    def reset(self):
        logging.info('resetting')
        self.set_current_state(self.init_state)
        return self.init_state

    def render(self):
        if self.display is None:
            self.display = Panda3dDisplay(self.physics)
        self.display.render_scene()

    def _wait_for_stability(self):
        logging.info('Waiting for walker to be stale')
        for i in range(MAX_STABILITY_STEPS):
            _, reward, _, _ = self.step()
            # self.render()
            if np.abs(reward) < MIN_MOVEMENT_FOR_STABILITY and i > 10:
                logging.info('Walker is stable')
                break

    def set_current_state(self, state):
        postions = state[: self.bone_count * 3].reshape([-1, 3])
        orientations = state[self.bone_count * 3: self.bone_count * 6].reshape([-1, 3])
        self.physics.set_bones_pos_hpr(postions, orientations)

    def get_current_state(self):
        return np.hstack((self.physics.get_bones_positions().flatten(),
                          self.physics.get_bones_orientations().flatten(),
                          self.physics.get_joint_angles()))

    def step(self, action=None):
        state = self.get_current_state()
        previous_walker_position = self.physics.get_walker_position()
        self.physics.apply_action(action)
        self.physics.step()
        walker_position = self.physics.get_walker_position()
        # reward = LA.norm((walker_position - previous_walker_position))
        reward = (walker_position - previous_walker_position)[0]
        # TODO: done should be reached after max distance from start
        # TODO: info should contain physical details (contact, velocity)
        info = None
        done = LA.norm((walker_position - previous_walker_position)) < MIN_MOVEMENT_FOR_END_EPISODE
        logging.debug('Step - Action: {}\n State: {}\n Reward: {:,.2f}\n Done: {}\n Info: {}\n'.format(
            action, state, reward, done, info))
        return state, reward, done, info
