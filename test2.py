
import math
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3,Vec4, TransformState, Point3
from panda3d.bullet import BulletWorld,  BulletHingeConstraint
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
import random
class Bone(object):
  def __init__(self):
    self.length = 0.5 + random.random()*2


class MyApp(ShowBase):
  def __init__(self):
    ShowBase.__init__(self)
    base.cam.setPos(0, -50, 20)
    base.cam.lookAt(0, 0, 0)
    self.N = 15
    self.INIT_HEIGHT = 10
    self.MASS = 0.5
    self.bones = []
    for i in range(self.N):
      self.bones.append(Bone())
    self.cs = []     
    # World
    self.world = BulletWorld()
    self.world.setGravity(Vec3(0, 0, -9.81))
    self.worldNP = render.attachNewNode('World')
    self.init_plain()
    self.add_box()
     
  def init_plain(self):
    # Plane
    shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
    node = BulletRigidBodyNode('Ground')
    node.addShape(shape)
    np = render.attachNewNode(node)
    np.setPos(0, 0, -2)
    self.world.attachRigidBody(node)

  def add_box(self):
    # Box
    shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
    nodeA = BulletRigidBodyNode()
    nodeA.setMass(self.MASS)
    nodeA.addShape(shape)

    npA = self.worldNP.attachNewNode(nodeA)
    npA.setPos(0, 0, self.INIT_HEIGHT)
    self.world.attachRigidBody(nodeA)
    model = loader.loadModel('models/box.egg')
    model.flattenLight()
    model.reparentTo(npA)
    for i, bone in enumerate(self.bones):
      shape2 = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
      nodeB = BulletRigidBodyNode()
      nodeB.setMass(0.5)
      nodeB.addShape(shape2)
      npB = self.worldNP.attachNewNode(nodeB)
      npB.setPos(i , 0, self.INIT_HEIGHT)
      npB.setScale(Vec3( 0.4, 0.4, bone.length))
      model = loader.loadModel('models/box.egg')
      self.world.attachRigidBody(nodeB)
      model.reparentTo(npB)
      frameA = TransformState.makePosHpr(Point3(0, 0, -0.1), Vec3(10* i, 90, -90))
      frameB = TransformState.makePosHpr(Point3(0, 0, bone.length), Vec3(10* i, 90, -90))
       
      self.cs.append(BulletHingeConstraint(npA.node(), npB.node(), frameA, frameB))
      self.cs[i].setLimit(-90, 90)
      self.world.attachConstraint(self.cs[i])
      npA = npB
    self.npB = npB
  # Update
  def update(self, task):
    dt = globalClock.getDt()
    # t = -self.npB.getHpr() * 5  
    for i in range(self.N):
      self.cs[i].enableAngularMotor(True, math.cos(task.time) * i, 1)

    self.world.doPhysics(dt)
    return task.cont
   
  def runy(self):
    taskMgr.add(self.update, 'update')
    base.run()

def main():
  MyApp().runy()
if __name__ == "__main__":
  main()