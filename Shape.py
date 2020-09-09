import random

INIT_Z = .5
BONES_COUNT = 4


class Joint(object):
    def __init__(self, parent_bone, child_bone, min_range=-90, max_range=90,
                 gap=None, child_start_hpr=(0, 0, 0), parent_start_hpr=(0, 0, 0)):
        self.child_start_hpr = child_start_hpr
        self.parent_start_hpr = parent_start_hpr
        self.angle_range = (min_range, max_range)
        self.parent_bone = parent_bone
        self.child_bone = child_bone
        self.gap_radius = parent_bone.height * 2 if gap is None else gap


class Bone(object):
    def __init__(self, index, start_pos, width=2, height=0.3, length=3, mass=1, start_hpr=(0, 0, 0), friction=0.75):
        self.name = 'Bone{}'.format(index)
        self.index = index
        self.width = width
        self.height = height
        self.length = length
        self.start_pos = start_pos
        self.start_hpr = start_hpr
        self.friction = friction
        self.mass = mass


class Shape(object):
    def __init__(self):
        self.bones = self._gen_bones()
        self.joints = self._gen_joints()

    def _gen_joints(self):
        return [Joint(self.bones[random.choice(range(i))], self.bones[i]) for i in range(1, len(self.bones))]


class Worm(Shape):
    def _gen_joints(self):
        return [Joint(self.bones[i - 1], self.bones[i]) for i in range(1, len(self.bones))]

    def _gen_bones(self):
        bones = []
        total_x = 0
        for i in range(BONES_COUNT):
            bones.append(Bone(index=i, start_pos=(total_x, 0, INIT_Z),
                              start_hpr=(0, 90, 0), friction=1.5))
            # start_hpr=(0, 90, 0), friction=(i + 1) * 5 / BONES_COUNT))
            # bones.append(Bone(index=i, start_pos=(total_x, 0, INIT_Z),
            # start_hpr=(0, 90, 0), friction=5 * int(i == BONES_COUNT - 1)))
            total_x += bones[-1].length * 2 + bones[-1].height
        return bones


class Legs(Shape):
    def _gen_bones(self):
        leg_diameter = 0.3
        thigh_calf_length = 2
        foot_diameter = 1.3
        foot_height = 0.1
        pelvic_diameter = 1.5
        self.pelvic = Bone(index=0, width=pelvic_diameter, height=pelvic_diameter,
                           length=pelvic_diameter, start_pos=(0, 0, 5.5), mass=5)
        self.left_thigh = Bone(index=1, width=leg_diameter, height=leg_diameter, start_hpr=(0, 0, 90),
                               length=thigh_calf_length, start_pos=(0, 2, 4.5), mass=1)
        self.right_thigh = Bone(index=4, width=leg_diameter, height=leg_diameter, start_hpr=(0, 0, 90),
                                length=thigh_calf_length, start_pos=(0, -2, 4.5), mass=1)
        self.left_calf = Bone(index=2, width=leg_diameter, height=leg_diameter, start_hpr=(0, 0, 90),
                              length=thigh_calf_length, start_pos=(0, 2, 2.5), mass=1)
        self.right_calf = Bone(index=5, width=leg_diameter, height=leg_diameter, start_hpr=(0, 0, 90),
                               length=thigh_calf_length, start_pos=(0, -2, 2.5), mass=1)
        self.left_foot = Bone(index=3, width=foot_diameter, height=foot_diameter, start_hpr=(0, 0, 90),
                              length=foot_height, start_pos=(0, 2, foot_height), mass=0.4)
        self.right_foot = Bone(index=6, width=foot_diameter, height=foot_diameter, start_hpr=(0, 0, 90),
                               length=foot_height, start_pos=(0, -2, foot_height), mass=0.4)
        bones = [self.right_foot, self.right_calf, self.right_thigh,
                 self.pelvic, self.left_thigh, self.left_calf, self.left_foot]
        return bones

    def _gen_joints(self):
        left_hip = Joint(self.pelvic, self.left_thigh, gap=self.left_thigh.height * 2,
                         parent_start_hpr=(0, 90, -90), min_range=-10)
        right_hip = Joint(self.right_thigh, self.pelvic, gap=self.left_thigh.height * 2,
                          child_start_hpr=(0, 90, 90), min_range=-10)
        left_knee = Joint(self.left_thigh, self.left_calf, max_range=0)
        right_knee = Joint(self.right_calf, self.right_thigh, max_range=0)
        left_ankle = Joint(self.left_calf, self.left_foot, min_range=-60, max_range=20,
                           gap=self.left_thigh.height)
        right_ankle = Joint(self.right_foot, self.right_calf, min_range=-60, max_range=20,
                            gap=self.left_thigh.height)
        return [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]
