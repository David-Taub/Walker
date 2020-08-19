import random

INIT_Z = 1
JOINT_GAP_RATIO = 1.1
BONE_MASS = 1
BONE_FRICTION = 2
BONES_COUNT = 5


class Joint(object):
    def __init__(self, parent_bone, child_bone):
        self.start_hpr = (0, 0, random.uniform(-60, 60))
        self.angle_range = (-120, 120)
        self.parent_bone = parent_bone
        self.child_bone = child_bone
        self.gap_radius = max(child_bone.width, child_bone.height, parent_bone.width,
                              parent_bone.height) * JOINT_GAP_RATIO


class Bone(object):
    def __init__(self, index, width=None, length=None, height=None):
        self.name = 'Bone{}'.format(index)
        self.index = index
        self.length = length if length is not None else random.uniform(1, 4)
        self.width = width if width is not None else random.uniform(0.2, 0.4)
        self.height = height if height is not None else random.uniform(0.2, 0.4)
        self.start_pos = (0, index * 10, INIT_Z)
        self.start_hpr = (0, 0, 0)
        self.friction = BONE_FRICTION
        self.mass = BONE_MASS


class Shape(object):

    def __init__(self):
        self.bones = self._gen_bones(BONES_COUNT)
        self.joints = self._gen_joints()

    def _gen_bones(self, amount):
        return [Bone(i) for i in range(amount)]

    def _gen_joints(self):
        return [Joint(self.bones[random.choice(range(i))], self.bones[i]) for i in range(1, len(self.bones))]

# class Spider(Shape):
#     def __init__(self):
#         body_size = 3
#         body_bones = [Bone(self._next_index()) for i in range(body_size)]
#         body_joints = [Joint(body_bones[i - 1], body_bones[i]) for i in range(1, body_size)]
#         for bone in body_bones:
#             bone.friction = 0.1
#         for joint in body_joints:
#             joint.power = joint.power * 10
#             joint.heading = 0
#             joint.roll = 0
#             joint.pitch = 0
#             joint.max_angle = 35
#             joint.min_angle = -35
#             joint.action_factor = 1

#         self.bones = body_bones
#         self.joints = body_joints
#         self.body = np.array(list(range(body_size)))
#         self.legs = []
#         self._add_leg(body_bones[0], 45)
#         self._add_leg(body_bones[0], -45)
#         # self._add_leg(body_bones[1], 45)
#         # self._add_leg(body_bones[1], -45)
#         self._add_leg(body_bones[2], 45)
#         self._add_leg(body_bones[2], -45)

#     def _add_leg(self, parent_bone, hip_heading, leg_size=3):
#         leg_bones = [Bone(self._next_index()) for i in range(leg_size)]
#         part = np.array([bone.index for bone in leg_bones])
#         leg_joints = [Joint(leg_bones[i - 1], leg_bones[i]) for i in range(1, leg_size)]
#         hip = Joint(parent_bone, leg_bones[0])
#         leg_joints.insert(0, hip)
#         for bone in leg_bones:
#             bone.length = 2.5
#             bone.friction = 1
#         for joint in leg_joints:
#             joint.heading = 0
#             joint.pitch = 0
#             joint.roll = 0
#         hip.roll = hip_heading
#         foot = leg_bones[-1]
#         foot.length = foot.height
#         foot.height, foot.width = 1, 1
#         foot.friction *= 100
#         ankle = leg_joints[-1]
#         ankle.max_angle = 20
#         ankle.min_angle = -20
#         self.bones += leg_bones
#         self.joints += leg_joints
#         self.legs.append(part)


class Worm(Shape):
    # def __init__(self):
    #     self.N = 4
    #     self.lengths = [5] * self.N
    #     self.widths = [0.5] * self.N
    #     self.heights = [0.5] * self.N
    #     self.masses = [3.5] * self.N
    #     self.frictions = [500] * self.N
    #     self.lengths[0] = 7
    #     self.widths[0] = 7
    #     self.masses[0] = 10
    #     self.heights[0] = 7
    #     self.frictions[0] = 0.3
    #     self.pitches = [0] * self.N
    #     self.positions = [[(i, i, INIT_Z), (0, 0, 0)] for i in self.lengths]
    #     super().__init__()

    def _gen_joints(self):
        return [Joint(self.bones[i - 1], self.bones[i]) for i in range(1, len(self.bones))]
