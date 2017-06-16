import random
class Walker(object):
  def __init__(self, app, N):
    self.app = app
    self.root = Bone(self.app)
    self.bones = [self.root]
    self.joints = []
    for i in range(self.app.N-1):
      child_bone = Bone(self.app)
      parent_bone = random.choice(self.bones)
      joint = Joint(parent_bone, child_bone, self.app)
      self.bones.append(child_bone)
      self.joints.append(joint)


class Joint(object):
  def __init__(self, parent_bone, child_bone, app):
      hpr = [random.uniform(-90,90),random.uniform(-90,90),random.uniform(-90,90)]
      pos = random.uniform(0,1)
      pos = 1
      self.constraint = app.add_joint(parent_bone, child_bone, hpr, pos)
      self.parent_bone = parent_bone
      self.child_bone = child_bone


class Bone(object):
  def __init__(self, app, length = None):
    self.WIDTH = 0.5
    if length is None:
      self.length = random.uniform(2.5, 4)
    else:
      self.length = length
    app.init_bone(self)
