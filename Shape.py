import random, math, sys, pickle, os
import numpy as np

class Shape(object):
    INIT_Z = 2
    MAX_IN_PART = 3
    def __init__(self, N):
        print("Generating random shape")
        self.start_score = 0
        self.N = N
        self.lengths = [random.uniform(1, 4) for i in range(N)]
        self.widths = [random.uniform(0.2, 0.4) for i in range(N)]
        self.heights = [random.uniform(0.2, 0.4) for i in range(N)]
        self.headings = [random.randrange(-180, 180) for i in range(N)]
        self.pitches = [random.randrange(-90, 90) for i in range(N)]
        self._gen_connections()
        self.positions = [(0, i, self.INIT_Z) for i in range(N)]

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
        if len(self.parts[pointers[parent]]) >= self.MAX_IN_PART:
            pointers.append(None)
        else:
            pointers.append(pointers[parent])
        pointers[parent] = None

