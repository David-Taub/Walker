
import random, math, sys, pickle, os
import numpy as np
import MyApp

class Shape(object):
  INIT_HEIGHT = 2
  def __init__(self, N):
      self.N = N
      self.lengths = [random.uniform(1, 4) for i in range(N)]  
      self.headings = [random.randrange(-180, 180) for i in range(N)]  
      self.pitches = [random.randrange(-90, 90) for i in range(N)]
      self.connections = [-1] 
      for i in range(N-1):
        c = random.randrange(i+1)
        while self.connections.count(c) == 2:
          c = random.randrange(i+1)
        self.connections.append(c)
      # self.connections = [-1] + [random.randrange(i+1) for i in range(N-1)]
      self.positions = [(0, i, self.INIT_HEIGHT) for i in range(N)]

class Walker(object):
  BUFFER_LENGTH = 0.5
  TIME_STEP = 0.1

  def __init__(self):
    self.app = MyApp.MyApp(self)
    self.last_score = None
    self.load_shape()
    

  def _get_pickle_path(self):
    return os.path.join(os.path.dirname(sys.argv[0]), 'shape.pickle')
  
  def reset(self):
    self.app.remove_shape()
    self.last_score = None
    self.load_shape()
    return self.get_state()

  def load_shape(self):
    pickle_path = self._get_pickle_path()
    if os.path.isfile(pickle_path):
      with open(self._get_pickle_path(), 'rb') as f:
        self.shape = pickle.load(f)
      self._build_bones_and_joints()
    else:
      self.shape = Shape(10)
      self._build_bones_and_joints()
      for i in range(300):
        self.step([0] * len(self.joints))
      self.save_shape()

  
  def _build_bones_and_joints(self):
    self.joints = []
    self.bones = []
    for i in range(self.shape.N):
        bone = Bone(self.app, self.shape.lengths[i], self.shape.positions[i], i)
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
    self.app.step(action, self.TIME_STEP)
    return self.get_state(), self.get_reward()

  def get_state(self):
    state = []
    state += self.app.get_joint_angles()
    state += self.app.get_bones_height()
    return np.array(state)

  def start(self, with_graphics, action_generator):
    self.action_generator = action_generator 
    self.app.runy(with_graphics)

  def gen_actions(self):
    return self.action_generator(self.get_state())

  def get_reward(self):
    if self.last_score is None:
      self.last_score = self.app.get_com()[2]
      return 0
    new = self.app.get_com()[2]
    ret = new - self.last_score
    self.last_score = new
    return ret

  def _gen_action(self, i):
    import time
    t = time.time()
    return  math.cos(t)

class Joint(object):
  def __init__(self, parent_bone, child_bone, app, heading, pitch):
      hpr = [heading, pitch, 0] 
      hpr = [0, pitch, 0] 
      pos = 1
      self.constraint = app.add_joint(parent_bone, child_bone, hpr, pos)
      self.parent_bone = parent_bone
      self.child_bone = child_bone


class Bone(object):
  def __init__(self, app, length, position, index):
    self.has_joint_ball = False
    self.width = 0.5
    self.length = length
    app.init_bone(self, position, index)

# Walker().start(True)