import random, math, sys, pickle, os

class Shape(object):
  INIT_Z = 2
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
    for i in range(N-1):
      c = random.randrange(i+1)
      while self.connections.count(c) == 2:
        c = random.randrange(i+1)
      self.connections.append(c)
    # self.connections = [-1] + [random.randrange(i+1) for i in range(N-1)]
    self.positions = [(0, i, self.INIT_Z) for i in range(N)]

