import random

INIT_Z = .5
BONE_MASS = 1
BONE_FRICTION = 1
BONES_COUNT = 6


class Joint(object):
    def __init__(self, parent_bone, child_bone):
        self.start_hpr = (0, 0, 0)
        self.angle_range = (-120, 120)
        self.parent_bone = parent_bone
        self.child_bone = child_bone
        self.gap_radius = parent_bone.height


class Bone(object):
    def __init__(self, index, start_y=0):
        self.name = 'Bone{}'.format(index)
        self.index = index
        self.length = 3
        self.width = 2
        self.height = 0.3
        self.start_pos = (start_y, 0, INIT_Z)
        self.start_hpr = (0, 90, 0)
        self.friction = BONE_FRICTION
        self.mass = BONE_MASS


class Shape(object):
    def __init__(self):
        self.bones = self._gen_bones(BONES_COUNT)
        self.joints = self._gen_joints()

    def _gen_bones(self, amount):
        bones = []
        total_y = 0
        for i in range(amount):
            bones.append(Bone(i, start_y=total_y))
            total_y += bones[-1].length * 2 + bones[-1].height
        return bones

    def _gen_joints(self):
        return [Joint(self.bones[random.choice(range(i))], self.bones[i]) for i in range(1, len(self.bones))]


class Worm(Shape):
    def _gen_joints(self):
        return [Joint(self.bones[i - 1], self.bones[i]) for i in range(1, len(self.bones))]
