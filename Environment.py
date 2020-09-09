import logging
from numpy import linalg as LA
from Panda3dPhysics import Panda3dPhysics
from Panda3dDisplay import Panda3dDisplay

import numpy as np




class Environment:

    JOINT_POWER = 3
    JOINT_SPEED = 2
    PLANE_FRICTION = 0.75
    GRAVITY_ACCELERATION = 9.81
    ANGLE_SCALE = 90
    PHYSICAL_STEPS_PER_ACTION = 10
    MAX_STEPS_PER_EPISODE = 500
    # stability on initialization
    MAX_STABILITY_STEPS = 500
    MIN_MOVEMENT_FOR_STABILITY = 0.0001

    # episode ending conditions
    LAST_VELOCITY_HISTORY_SIZE = 40
    LAST_VELOCITY_AVERAGE_INIT = 1
    MIN_MOVEMENT_FOR_END_EPISODE = 0.1

    # reward constants

    # larger value causes longer episodes
    TIME_STEP_REWARD = 0.05
    # larger value causes faster walking
    VELOCITY_REWARD = 10
    # larger value causes smoother walking
    VELOCITY_DECREASE_PENALTY = 0.01
    # larger value causes straighter walking
    SIDE_PROGRESS_PENALTY = 1.5
    # larger value causes less unneeded movements
    ACTUATOR_PENALTY = 0.05
    # larger value causes less unneeded movements
    OVER_PRESS_JOINT_PENALTY = 0.5

    def __init__(self, walker, render=False):
        self.physics = Panda3dPhysics(joint_power=3, joint_speed=2, plane_friction=0.75, gravity_acceleration=9.81)
        self.physics.add_walker(walker)
        self.display = Panda3dDisplay(self.physics)
        self.close_window()
        self._wait_for_stability(render)
        self.init_state = self.get_current_state()
        self.state_size = len(self.init_state)
        self.action_size = len(self.physics.constraints)
        self.init_bones_positions = self.physics.get_bones_relative_positions()
        self.init_bones_orientations = self.physics.get_bones_orientations()
        self.reset()

    def open_window(self):
        logging.debug('Opening window')
        self.display = Panda3dDisplay(self.physics)

    def close_window(self):
        logging.debug('Closing window')
        self.display.close_window()

    def reset(self):
        logging.debug('Resetting environment')
        self.episode_reward = 0
        self.step_index = 0
        positions = self.init_bones_positions
        positions[:, 2] -= np.min(positions[:, 2])
        orientations = self.init_bones_orientations
        self.physics.set_bones_pos_hpr(positions, orientations)
        self.last_velocity = [self.LAST_VELOCITY_AVERAGE_INIT] * self.LAST_VELOCITY_HISTORY_SIZE
        return self.get_current_state()

    def render(self):
        self.display.render_scene()

    def _wait_for_stability(self, render):
        logging.debug('Waiting for walker to be stale')
        for i in range(self.MAX_STABILITY_STEPS):
            last_pos = self.physics.get_walker_position()
            self.physics.step()
            movement = LA.norm(last_pos - self.physics.get_walker_position())
            if render:
                self.render()
            # time.sleep(1.5)
            if np.abs(movement) < self.MIN_MOVEMENT_FOR_STABILITY and i > 10:
                logging.debug('Walker is stable')
                break

    def get_current_state(self):
        # logging.info(
        #     'get_bones_relative_positions: ' + str(self.physics.get_bones_relative_positions().flatten()) + '\n' +
        #     'get_bones_linear_velocity: ' + str(self.physics.get_bones_linear_velocity().flatten()) + '\n' +
        #     'get_bones_orientations: ' + str(self.physics.get_bones_orientations().flatten() / self.ANGLE_SCALE) + '\n' +
        #     'get_bones_angular_velocity: ' + str(self.physics.get_bones_angular_velocity().flatten() / self.ANGLE_SCALE) + '\n' +
        #     'get_joint_angles: ' + str(self.physics.get_joint_angles() / self.ANGLE_SCALE) + '\n' +
        #     'get_joint_angles_diff: ' + str(self.physics.get_joint_angles_diff()) + '\n' +
        #     'get_bones_ground_contacts: ' + str(self.physics.get_bones_ground_contacts()) + '\n' +
        #     'prev_action: ' + str(self.physics.prev_action) + '\n')

        return np.hstack((self.physics.get_bones_relative_positions().flatten(),
                          self.physics.get_bones_linear_velocity().flatten(),
                          self.physics.get_bones_orientations().flatten() / self.ANGLE_SCALE,
                          self.physics.get_bones_angular_velocity().flatten() / self.ANGLE_SCALE,
                          self.physics.get_joint_angles() / self.ANGLE_SCALE,
                          self.physics.get_joint_angles_diff() / self.ANGLE_SCALE,
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

    def step(self, action):
        self.step_index += 1
        previous_velocity = self.get_walker_x_velocity()
        self.physics.apply_action(action)
        for i in range(self.PHYSICAL_STEPS_PER_ACTION):
            self.physics.step()
        state = self.get_current_state()
        reward = self.VELOCITY_REWARD * self.get_walker_x_velocity()
        reward += self.TIME_STEP_REWARD * self.step_index / self.MAX_STEPS_PER_EPISODE
        reward -= self.VELOCITY_DECREASE_PENALTY * max(0, previous_velocity - self.get_walker_x_velocity()) ** 2
        reward -= self.SIDE_PROGRESS_PENALTY * self.physics.get_walker_position()[1] ** 2
        reward -= self.OVER_PRESS_JOINT_PENALTY * np.mean((
            action * self.physics.get_joint_angles() / self.ANGLE_SCALE) ** 4)

        reward -= self.ACTUATOR_PENALTY * np.mean(action ** 2)
        self.episode_reward += reward
        # done episode if walker is stuck, low average velocity
        done = self.get_walker_x_velocity() == 0 or self.step_index > self.MAX_STEPS_PER_EPISODE or \
            self.episode_reward < -30 or \
            self.update_last_velocity_average() < self.MIN_MOVEMENT_FOR_END_EPISODE
        info = None
        # logging.info('Step {}- Action: {}\n State: {}\n Reward: {:,.2f}\n Done: {}\n Info: {}\n'.format(self.step_index,
                                                                                                        # action, state, reward, done, info))
        return state, reward, done, info
