import random
import numpy as np

INIT_Z = 3
MAX_IN_PART = 4
NUM_OF_BONES = 8


class Joint(object):
    def __init__(self, parent_bone, child_bone):
        self.power = 100
        self.action_factor = 2
        self.pitch = 0
        self.heading = 0
        self.roll = random.randrange(-90, 90)
        self.min_angle = -90
        self.max_angle = 0
        self.parent_bone = parent_bone
        self.child_bone = child_bone
        self.gap_radius = max(child_bone.width, child_bone.height, parent_bone.width, parent_bone.height) * 1.1


class Bone(object):
    def __init__(self, index):
        self.index = index
        self.length = random.uniform(1, 4)
        self.width = random.uniform(0.2, 0.4)
        self.height = random.uniform(0.2, 0.4)
        self.start_pos = (0, index * 10, INIT_Z)
        self.start_hpr = (0, 0, 0)
        self.mass = 1
        self.friction = 3


class Shape(object):
    def __init__(self):
        self.bones = [Bone(self._next_index()) for i in range(2)]
        connections = self._gen_connections()
        self.joints = [Joint(self.bones[connections[i]], self.bones[i]) for i in range(1, len(connections))]

    def _next_index(self):
        if not hasattr(self, 'current_index'):
            self.current_index = 0
        self.current_index += 1
        return self.current_index - 1

    def _gen_connections(self):
        connections = [-1]
        self.parts = [np.array([0])]
        pointers = [0]

        for child in range(1, len(self.bones)):
            parent = random.randrange(child)
            while connections.count(parent) == 2:
                # TODO: remove this while, it is ugly
                parent = random.randrange(child)
            connections.append(parent)
            self._update_parts(parent, child, pointers)
        return connections

    def _update_parts(self, parent, child, pointers):
        if pointers[parent] is None:
            self.parts.append(np.array([parent, child]))
            pointers.append(len(self.parts) - 1)
            return
        self.parts[pointers[parent]] = np.append(self.parts[pointers[parent]], child)
        if len(self.parts[pointers[parent]]) >= MAX_IN_PART:
            pointers.append(None)
        else:
            pointers.append(pointers[parent])
        pointers[parent] = None


class Spider(Shape):
    def __init__(self):
        body_size = 3
        body_bones = [Bone(self._next_index()) for i in range(body_size)]
        body_joints = [Joint(body_bones[i - 1], body_bones[i]) for i in range(1, body_size)]
        for bone in body_bones:
            bone.friction = 0.1
        for joint in body_joints:
            joint.power = joint.power * 10
            joint.heading = 0
            joint.roll = 0
            joint.pitch = 0
            joint.max_angle = 35
            joint.min_angle = -35
            joint.action_factor = 1

        self.bones = body_bones
        self.joints = body_joints
        self.body = np.array(list(range(body_size)))
        self.legs = []
        self._add_leg(body_bones[0], 45)
        self._add_leg(body_bones[0], -45)
        # self._add_leg(body_bones[1], 45)
        # self._add_leg(body_bones[1], -45)
        self._add_leg(body_bones[2], 45)
        self._add_leg(body_bones[2], -45)

    def _add_leg(self, parent_bone, hip_heading, leg_size=3):
        leg_bones = [Bone(self._next_index()) for i in range(leg_size)]
        part = np.array([bone.index for bone in leg_bones])
        leg_joints = [Joint(leg_bones[i - 1], leg_bones[i]) for i in range(1, leg_size)]
        hip = Joint(parent_bone, leg_bones[0])
        leg_joints.insert(0, hip)
        for bone in leg_bones:
            bone.length = 2.5
            bone.friction = 1
        for joint in leg_joints:
            joint.heading = 0
            joint.pitch = 0
            joint.roll = 0
        hip.roll = hip_heading
        foot = leg_bones[-1]
        foot.length = foot.height
        foot.height, foot.width = 1, 1
        foot.friction *= 100
        ankle = leg_joints[-1]
        ankle.max_angle = 20
        ankle.min_angle = -20
        self.bones += leg_bones
        self.joints += leg_joints
        self.legs.append(part)


# class WormShape(Shape):
#     def __init__(self):
#         print("Generating worm shape")
#         self.N = 4
#         self._init_bones_traits()
#         self._gen_connections()

#     def _init_bones_traits(self):
#         self.lengths = [5] * self.N
#         self.widths = [0.5] * self.N
#         self.heights = [0.5] * self.N
#         self.masses = [3.5] * self.N
#         self.frictions = [500] * self.N
#         self.lengths[0] = 7
#         self.widths[0] = 7
#         self.masses[0] = 10
#         self.heights[0] = 7
#         self.frictions[0] = 0.3
#         self.pitches = [0] * self.N
#         self.positions = [[(i, i, INIT_Z), (0,0,0) ] for i in self.lengths]

#     def _gen_connections(self):
#         self.connections = list(range(-1, self.N-1))
#         self.parts = [np.array(range(self.N))]
