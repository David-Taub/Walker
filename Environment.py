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
    def reset():
        return state

    def render():

    def step(action):
        return state, reward, done, info
