import logging
import Shape
from numpy import linalg as LA
from Panda3dPhysics import Panda3dPhysics
from Panda3dDisplay import Panda3dDisplay

import numpy as np

MAX_STABILITY_STEPS = 500
MIN_MOVEMENT_FOR_STABILITY = 0.0001
MIN_MOVEMENT_FOR_END_EPISODE = 0.001
PHYSICAL_STEPS_PER_ACTION = 10
MAX_SCORE = 100
STUCK_PENALTY = -100


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
        self.reset()
        self.init_position = self.physics.get_walker_position()

    def reset(self):
        logging.debug('resetting')
        postions = self.init_state[: self.bone_count * 3].reshape([-1, 3])
        postions[:, 2] -= np.min(postions[:, 2])
        orientations = self.init_state[self.bone_count * 3: self.bone_count * 6].reshape([-1, 3])
        self.physics.set_bones_pos_hpr(postions, orientations)
        return self.init_state

    def render(self):
        if self.display is None:
            self.display = Panda3dDisplay(self.physics)
        self.display.render_scene()

    def _wait_for_stability(self):
        logging.debug('Waiting for walker to be stale')
        for i in range(MAX_STABILITY_STEPS):
            last_pos = self.physics.get_walker_position()
            self.physics.step()
            movement = LA.norm(last_pos - self.physics.get_walker_position())
            self.render()
            if np.abs(movement) < MIN_MOVEMENT_FOR_STABILITY and i > 10:
                logging.debug('Walker is stable')
                break

    def get_current_state(self):
        return np.hstack((self.physics.get_bones_relative_positions().flatten(),
                          self.physics.get_bones_orientations().flatten(),
                          self.physics.get_joint_angles()))

    def get_score(self):
        return self.physics.get_walker_position()[0] - self.init_position[0]

    def step(self, action=None):
        state = self.get_current_state()
        previous_walker_position = self.physics.get_walker_position()
        self.physics.apply_action(action)
        for i in range(PHYSICAL_STEPS_PER_ACTION):
            self.physics.step()
        walker_position = self.physics.get_walker_position()
        is_stuck = LA.norm((walker_position - previous_walker_position)) < MIN_MOVEMENT_FOR_END_EPISODE
        max_score_achieved = self.get_score() > MAX_SCORE
        done = max_score_achieved or is_stuck
        # reward = LA.norm((walker_position - previous_walker_position))
        reward = (walker_position - previous_walker_position)[0]
        if is_stuck:
            reward = STUCK_PENALTY
        info = None
        logging.debug('Step - Action: {}\n State: {}\n Reward: {:,.2f}\n Done: {}\n Info: {}\n'.format(
            action, state, reward, done, info))
        return state, reward, done, info
