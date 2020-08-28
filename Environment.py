import logging
import Shape
from numpy import linalg as LA
from Panda3dPhysics import Panda3dPhysics
from Panda3dDisplay import Panda3dDisplay

import numpy as np
import time

MAX_STABILITY_STEPS = 500
MIN_MOVEMENT_FOR_STABILITY = 0.0001
MIN_MOVEMENT_FOR_END_EPISODE = 0.0001
PHYSICAL_STEPS_PER_ACTION = 5
STUCK_PENALTY = -1
TIME_STEP_REWARD = 0.05
LAST_VELOCITY_HISTORY_SIZE = 80
LAST_VELOCITY_AVERAGE_INIT = 1
MIN_LAST_VELOCITY_AVERAGE = 0.5
ACTUATOR_PENALTY = 1
VELOCITY_REWARD = 5


class Environment:
    def __init__(self, walker, render=False):
        self.display = None
        self.physics = Panda3dPhysics()
        self.physics.add_walker(walker)
        self._wait_for_stability(render)
        self.init_state = self.get_current_state()
        self.state_size = len(self.init_state)
        self.action_size = len(self.physics.constraints)
        self.init_bones_positions = self.physics.get_bones_relative_positions()
        self.init_bones_orientations = self.physics.get_bones_orientations()
        self.reset()

    def reset(self):
        logging.debug('resetting')
        positions = self.init_bones_positions
        positions[:, 2] -= np.min(positions[:, 2])
        orientations = self.init_bones_orientations
        self.physics.set_bones_pos_hpr(positions, orientations)
        self.last_velocity = [LAST_VELOCITY_AVERAGE_INIT] * LAST_VELOCITY_HISTORY_SIZE
        return self.get_current_state()

    def render(self):
        if self.display is None:
            self.display = Panda3dDisplay(self.physics)
        self.display.render_scene()

    def _wait_for_stability(self, render):
        logging.debug('Waiting for walker to be stale')
        for i in range(MAX_STABILITY_STEPS):
            last_pos = self.physics.get_walker_position()
            self.physics.step()
            movement = LA.norm(last_pos - self.physics.get_walker_position())
            if render:
                self.render()
            # time.sleep(1.5)
            if np.abs(movement) < MIN_MOVEMENT_FOR_STABILITY and i > 10:
                logging.debug('Walker is stable')
                break

    def get_current_state(self):
        # logging.info(
        #     'get_bones_relative_positions: ' + str(self.physics.get_bones_relative_positions().flatten()) + '\n' +
        #     'get_bones_linear_velocity: ' + str(self.physics.get_bones_linear_velocity().flatten()) + '\n' +
        #     'get_bones_orientations: ' + str(self.physics.get_bones_orientations().flatten() / 180) + '\n' +
        #     'get_bones_angular_velocity: ' + str(self.physics.get_bones_angular_velocity().flatten() / 180) + '\n' +
        #     'get_joint_angles: ' + str(self.physics.get_joint_angles() / 180) + '\n' +
        #     'get_joint_angles_diff: ' + str(self.physics.get_joint_angles_diff()) + '\n' +
        #     'get_bones_ground_contacts: ' + str(self.physics.get_bones_ground_contacts()) + '\n' +
        #     'prev_action: ' + str(self.physics.prev_action) + '\n')

        return np.hstack((self.physics.get_bones_relative_positions().flatten(),
                          self.physics.get_bones_linear_velocity().flatten(),
                          self.physics.get_bones_orientations().flatten() / 180,
                          self.physics.get_bones_angular_velocity().flatten() / 180,
                          self.physics.get_joint_angles() / 180,
                          self.physics.get_joint_angles_diff() / 180,
                          self.physics.get_bones_ground_contacts(),
                          self.physics.prev_action
                          ))

    def get_score(self):
        return self.physics.get_walker_position()[0]

    def get_walker_x_velocity(self):
        return np.mean([velocity[0] for velocity in self.physics.get_bones_linear_velocity()])

    def update_last_velocity_average(self):
        self.last_velocity.pop(0)
        self.last_velocity.append(self.get_walker_x_velocity())
        return np.mean(self.last_velocity)

    def step(self, action=None):
        assert not np.isnan(np.sum(action))
        self.physics.apply_action(action)
        for i in range(PHYSICAL_STEPS_PER_ACTION):
            self.physics.step()
        state = self.get_current_state()
        assert not np.isnan(np.sum(state))
        # done episode if walker is stuck, low average velocity
        done = self.update_last_velocity_average() < MIN_MOVEMENT_FOR_END_EPISODE
        reward = VELOCITY_REWARD * self.get_walker_x_velocity()
        reward += TIME_STEP_REWARD
        if action is not None:
            reward -= ACTUATOR_PENALTY * np.mean(action ** 2)
        info = None
        # logging.info('Step - Action: {}\n State: {}\n Reward: {:,.2f}\n Done: {}\n Info: {}\n'.format(
        # action, state, reward, done, info))
        return state, reward, done, info
