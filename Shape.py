import random, math, sys, pickle, os
import numpy as np

INIT_Z = 50
MAX_IN_PART = 4
NUM_OF_BONES = 3
class Shape(object):
    def __init__(self):
        print("Generating random shape")
        self.start_score = 0
        self.N = NUM_OF_BONES
        self._init_bones_traits()
        self._gen_connections()


    def _init_bones_traits(self):
        self.lengths = [random.uniform(1, 4) for i in range(self.N)]
        self.widths = [random.uniform(0.2, 0.4) for i in range(self.N)]
        self.heights = [random.uniform(0.2, 0.4) for i in range(self.N)]
        self.pitches = [random.randrange(-90, 90) for i in range(self.N)]
        self.positions = [[(0, i, INIT_Z), (0,0,90) ] for i in range(self.N)]

    def _gen_connections(self):
        self.connections = [-1]
        self.parts = [np.array([0])]
        pointers = [0]

        for child in range(1, self.N):
            parent = random.randrange(child)
            while self.connections.count(parent) == 2:
                #TODO: remove this while, it is ugly
                parent = random.randrange(child)
            self.connections.append(parent)
            self._update_parts(parent, child, pointers)


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

