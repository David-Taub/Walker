import random, math, sys, pickle, os
import numpy as np

INIT_Z = 3
MAX_IN_PART = 4
NUM_OF_BONES = 8

class Joint(object):
    def __init__(self, parent_bone, child_bone):
        self.pitch      = random.randrange(-90, 90)
        self.heading    = 0
        self.range      = 0
        self.min_angle  = -90
        self.max_angle  = 0
        self.parent_bone = parent_bone
        self.child_bone = child_bone
        self.gap_radius = max(child_bone.width, child_bone.height, parent_bone.width, parent_bone.height)


class Bone(object):
    def __init__(self, index):
        self.index      = index
        self.length     = random.uniform(1, 4)
        self.width      = random.uniform(0.2, 0.4)
        self.height     = random.uniform(0.2, 0.4)
        self.pitch      = random.randrange(-90, 90)
        self.start_pos  = (0, index*10, INIT_Z)
        self.start_hpr  = (0,0,90)
        self.mass       = 1
        self.friction   = 3

class Shape(object):
    def __init__(self):
        print("Generating random shape")

        self.start_score = 0
        self.N = NUM_OF_BONES
        self.bones = [Bone(i) for i in range(self.N)]
        connections = self._gen_connections()
        self.joints = [Joint(self.bones[connections[i]], self.bones[i]) for i in range(1,len(connections))]


    def _build_bones_and_joints(self):
        for i in range(self.shape.N):
            self.app.init_bone(bone)
            self.bones.append(bone)
            if self.shape.connections[i] != -1:
                self.app.add_joint(joint)
                self.joints.append(joint)


    def _gen_connections(self):
        connections = [-1]
        self.parts = [np.array([0])]
        pointers = [0]

        for child in range(1, self.N):
            parent = random.randrange(child)
            while connections.count(parent) == 2:
                #TODO: remove this while, it is ugly
                parent = random.randrange(child)
            connections.append(parent)
            self._update_parts(parent, child, pointers)
        return connections

    def _update_parts(self, parent, child, pointers):
        if pointers[parent] == None:
            self.parts.append(np.array([parent, child]))
            pointers.append(len(self.parts) - 1)
            return
        self.parts[pointers[parent]] = np.append(self.parts[pointers[parent]] , child)
        if len(self.parts[pointers[parent]]) >= MAX_IN_PART:
            pointers.append(None)
        else:
            pointers.append(pointers[parent])
        pointers[parent] = None

class WormShape(Shape):
    def __init__(self):
        print("Generating worm shape")
        self.start_score = 0
        self.N = 4
        self._init_bones_traits()
        self._gen_connections()

    def _init_bones_traits(self):
        self.lengths = [5] * self.N
        self.widths = [0.5] * self.N
        self.heights = [0.5] * self.N
        self.masses = [3.5] * self.N
        self.frictions = [500] * self.N
        self.lengths[0] = 7
        self.widths[0] = 7
        self.masses[0] = 10
        self.heights[0] = 7
        self.frictions[0] = 0.3
        self.pitches = [0] * self.N
        self.positions = [[(i, i, INIT_Z), (0,0,0) ] for i in self.lengths]

    def _gen_connections(self):
        self.connections = list(range(-1, self.N-1))
        self.parts = [np.array(range(self.N))]
