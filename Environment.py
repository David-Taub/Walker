import Shape
import logging
import sys
import pickle
import os

import numpy as np
from numpy import linalg as LA
from Panda3dPhysics import Panda3dPhysics
from Panda3dDisplay import Panda3dDisplay
import Shape

PHISICAL_STEP_TIME = 0.1
PHYSICAL_STEPS_IN_STEP = 1
MAX_STABILITY_STEPS = 100
MIN_MOVEMENT_FOR_STABILITY = 0.1


class Environment:
    def __init__(self):
        self.physics = Panda3dPhysics()
        walker = Shape.Worm()
        self.bone_count = len(walker.bones)
        self.physics.add_walker(walker)
        self._wait_for_stability()
        self.init_state = self.get_current_state()
        self.display = None

    def reset(self):
        self.set_current_state(self.init_state)
        return self.init_state

    def render(self):
        if self.display is None:
            self.display = Panda3dDisplay()
        self.display.render2()

    def _wait_for_stability(self):
        for i in range(MAX_STABILITY_STEPS):
            _, reward, _, _ = self.step()
            if reward < MIN_MOVEMENT_FOR_STABILITY:
                break

    def set_current_state(self, state):
        postions = self.init_state[: self.bone_count * 3].reshape([-1, 3])
        orientations = self.init_state[self.bone_count * 3: self.bone_count * 6].reshape([-1, 3])
        self.physics.set_bones_positions(postions)
        self.physics.set_bones_orientations(orientations)

    def get_current_state(self):
        return np.hstack((self.physics.get_bones_relative_positions().flatten(),
                          self.physics.get_bones_orientations().flatten(),
                          self.physics.get_joint_angles()))

    def step(self, action=None):
        state = self.get_current_state()
        previous_walker_position = self.physics.get_walker_position()
        self.physics.apply_action(action)
        self.physics.step()
        walker_position = self.physics.get_walker_position()
        reward = LA.norm((walker_position - previous_walker_position))
        # TODO: done should be reached after max distance from start
        # TODO: info should contain physical details (contact, velocity)
        info = None
        done = False
        return state, reward, done, info
