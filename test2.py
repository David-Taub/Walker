from direct.task import Task
import math, random
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from panda3d.bullet import *
from panda3d.bullet import BulletWorld,  BulletHingeConstraint
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
import random
from  Walker import Walker, Bone, Joint
class Bone(object):
  def __init__(self):
    self.length = random.uniform(0.5, 2)


class MyApp(ShowBase):
  def __init__(self):
    ShowBase.__init__(self)
    base.cam.setPos(0, -50, 20)
    base.cam.lookAt(0, 0, 0)
    self.N = 10
    self.c = 0
    self.INIT_HEIGHT = 10
    self.MASS = 1
    # World
    self.world = BulletWorld()
    self.world.setGravity(Vec3(0, 0, -9.81))
    self.worldNP = render.attachNewNode('World')
    self.init_plane()
    self.add_light()
    self.add_debug()
    self.walker = Walker(self, self.N)

  def add_debug(self):
    debugNode = BulletDebugNode('Debug')
    debugNode.showWireframe(False)
    debugNode.showConstraints(True)
    debugNode.showBoundingBoxes(False)
    debugNode.showNormals(True)
    debugNP = render.attachNewNode(debugNode)
    debugNP.show()
    self.world.setDebugNode(debugNP.node())
  
  def add_light(self):
    ambientLight = AmbientLight('ambientLight')
    ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
    ambientLightNP = render.attachNewNode(ambientLight)
    render.setLight(ambientLightNP)
    # SpotLight
    slight = render.attachNewNode(Spotlight("Spot"))
    slight.setPos(0,80,30)
    slight.lookAt(0,0,0)
    slight.node().setShadowCaster(True, 1024, 1024)
    slight.node().getLens().setFov(90)
    render.setLight(slight)
    slight = render.attachNewNode(Spotlight("Spot2"))
    slight.setPos(80,0,30)
    slight.lookAt(0,0,0)
    slight.node().setShadowCaster(True, 1024, 1024)
    slight.node().getLens().setFov(90)
    render.setLight(slight)
    

    render.setShaderAuto()
    
     
  def init_plane(self):
    # Plane
    shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
    node = BulletRigidBodyNode('Ground')
    node.addShape(shape)
    node.setFriction(1)
    np = render.attachNewNode(node)
    np.setPos(0, 0, -1)
    self.world.attachRigidBody(node)
    self.plane = Plane(Vec3(0, 0, 0), Point3(0, 0, 0))
    cm = CardMaker('')
    cm.setFrame(-100, 100, -100, 100)
    node = cm.generate()
    card = render.attachNewNode(node)
    card.setPos(0, 0, 0)
    card.lookAt(0, 0, -1)
    tex = loader.loadTexture('maps/grid.rgb')
    card.setTexture(tex)
    card.setShaderAuto()



  def init_bone(self, bone):
    # Box
    shape = BulletBoxShape(Vec3(bone.length, bone.WIDTH, bone.WIDTH))
    ts = TransformState.makePos(Point3(bone.length, bone.WIDTH, bone.WIDTH))
    node = BulletRigidBodyNode('Bone%d' % self.c)
    node.setMass(self.MASS)
    node.setFriction(1)
    node.addShape(shape, ts)

    bone.np = render.attachNewNode(node)
    # bone.np.setScale(Vec3(bone.length, 1, 1))
    bone.np.setPos(0, self.c, self.INIT_HEIGHT)
    bone.np.setShaderAuto()
    self.c += 0
    self.world.attachRigidBody(node)
    model = loader.loadModel('models/box.egg')
    model.setScale(Vec3(2*bone.length, 1, 1))
    model.reparentTo(bone.np)

  def add_joint(self, parent_bone, child_bone, hpr, pos):
    BUFFER_LENGTH = 0.7
    parent_frame = TransformState.makePosHpr(Vec3(parent_bone.length * 2 * pos +BUFFER_LENGTH, parent_bone.WIDTH,parent_bone.WIDTH), Vec3(90,0,0))
    # parent_frame = parent_frame.setScale( Vec3(parent_bone.length, 1, 1))
    # child_frame = TransformState.makePosHpr(Vec3(2 * child_bone.length + BUFFER_LENGTH,Bone.WIDTH,Bone.WIDTH), Vec3(*hpr))
    child_frame = TransformState.makePosHpr(Vec3(- BUFFER_LENGTH, child_bone.WIDTH,child_bone.WIDTH), Vec3(*hpr))
    # child_frame = child_frame.setScale(Vec3(child_bone.length, 1, 1))
    constraint = BulletHingeConstraint(parent_bone.np.node(), child_bone.np.node(), parent_frame, child_frame)
    constraint.setLimit(-70, 70)
    constraint.enableFeedback(True)
    constraint.setDebugDrawSize(2.0)
    self.world.attachConstraint(constraint)
    return constraint
  def get_state(self):
    state = []
    for i in range(len(self.walker.bones)):
      state.append(self.walker.bones[i].np.getPos()[2])
    for i in range(len(self.walker.joints)):
      state.append(self.walker.joints[i].constraint.getHingeAngle())
    return state

  def apply_action(self, action):
    for i in range(len(self.walker.joints)):
      self.walker.joints[i].constraint.enableAngularMotor(True, action[i], 1)
    
  def update(self, task):
    dt = globalClock.getDt()
    state = self.get_state()
    action = self.get_action(state)
    self.apply_action(action)
    self.world.doPhysics(dt)
    return task.cont

  def get_action(self, state):
    return [100*math.cos(globalClock.getFrameTime())] * self.N
  
  def spinCameraTask(self, task):
      angleDegrees = task.time * 6.0
      angleRadians = angleDegrees * (math.pi / 180.0)
      self.camera.setPos(20 * math.sin(angleRadians), -20.0 * math.cos(angleRadians), 3)
      self.camera.setHpr(angleDegrees, 0, 0)
      return Task.cont
   
  def runy(self):
    taskMgr.add(self.update, 'update')
    taskMgr.add(self.spinCameraTask, "SpinCameraTask")
    base.run()

def main():
  MyApp().runy()
if __name__ == "__main__":
  main()