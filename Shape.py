import random, math, sys, pickle, os

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

        self.connections = [-1]
        self.parts = [[0]]
        pointers = [0]

        for child in range(N-1):
            parent = random.randrange(child+1)
            while self.connections.count(parent) == 2:
                #TODO: remove this while, it is ugly
                parent = random.randrange(child+1)
            self.connections.append(parent)
            self._update_parts(parent, child, pointers)
        self.positions = [(0, i, self.INIT_Z) for i in range(N)]

    def _update_parts(self, parent, child, pointers):
        if pointers[parent] == None:
            self.parts.append([parent, child])
            pointers.append(len(parts) - 1)
            return
        self.parts[pointers[parent]].append(child)
        if len(self.parts[pointers[parent]]) >= self.MAX_IN_PART:
            pointers.append(None)
        else:
            pointers.append(pointers[parent])
        pointers[parent] = None

