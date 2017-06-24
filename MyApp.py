from direct.task import Task
import math, random
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from panda3d.bullet import *
from panda3d.bullet import BulletWorld,  BulletHingeConstraint
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape


class MyApp(ShowBase):
  def __init__(self, walker):
    ShowBase.__init__(self)
    base.cam.setPos(0, -50, 20)
    base.cam.lookAt(0, 0, 0)
    self.c = 0
    self.MASS = 1
    # World
    self.world = BulletWorld()
    self.world.setGravity(Vec3(0, 0, -9.81))
    self.worldNP = render.attachNewNode('World')
    self.init_plane()
    self.add_light()
    # self.add_ ()
    self.walker = walker

  def add_debug(self):
    debugNode = BulletDebugNode('Debug')
    debugNode.showWireframe(True)
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
    self.tracker_light = render.attachNewNode(Spotlight("tracker"))
    self.tracker_light.setPos(0,80,30)
    self.tracker_light.lookAt(0,0,0)
    self.tracker_light.node().setShadowCaster(True, 1024, 1024)
    self.tracker_light.node().getLens().setFov(90)
    render.setLight(self.tracker_light)
    slight = render.attachNewNode(Spotlight("Spot2"))
    slight.setPos(80,0,30)
    slight.lookAt(0,0,0)
    slight.node().setShadowCaster(True, 1024, 1024)
    slight.node().getLens().setFov(90)
    render.setLight(slight)
    render.setShaderAuto()

  def light_track(self):
    self.tracker_light.setPos(self.get_com()+Vec3(0,80,30))


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
    cm.setFrame(-1000, 1000, -1000, 1000)
    node = cm.generate()
    card = render.attachNewNode(node)
    card.setPos(0, 0, 0)
    card.lookAt(0, 0, -1)
    tex = loader.loadTexture('maps/grid.rgb')
    card.setTexture(tex)


  def get_bones_positions(self):
    return [[bone.np.getPos(), bone.np.getHpr()] for bone in self.walker.bones]

  def init_bone(self, bone, position, index):
    # Box
    shape = BulletBoxShape(Vec3(bone.length, bone.width, bone.width))
    ts = TransformState.makePos(Point3(bone.length, bone.width, bone.width))
    node = BulletRigidBodyNode('Bone%d' % index)
    node.setMass(self.MASS)
    node.setFriction(1)
    node.addShape(shape, ts)

    bone.np = render.attachNewNode(node)
    bone.np.setPos(position[0])
    bone.np.setHpr(position[1])
    bone.np.setShaderAuto()
    self.world.attachRigidBody(node)
    bone.model = loader.loadModel('models/box.egg')
    bone.model.setScale(Vec3(2*bone.length, 2*bone.width, 2*bone.width))
    bone.model.reparentTo(bone.np)
    bone.np.setColor(0.6, 0.6, 1.0, 1.0)

  def add_joint_ball(self, bone):
    if bone.has_joint_ball:
      return
    model = loader.loadModel('smiley.egg')
    model.reparentTo(render)
    scale_vec = Vec3(self.walker.BUFFER_LENGTH,self.walker.BUFFER_LENGTH,self.walker.BUFFER_LENGTH)
    model.setTransform(TransformState.makePos(Point3(bone.length * 2 + self.walker.BUFFER_LENGTH, bone.width, bone.width)))
    model.setScale(scale_vec)
    tex = loader.loadTexture('maps/noise.rgb')
    model.setTexture(tex, 1)
    model.reparentTo(bone.np)

    bone.has_joint_ball = True

  def add_joint(self, parent_bone, child_bone, hpr, pos):
    self.add_joint_ball(parent_bone)
    parent_frame_pos = Vec3(parent_bone.length * 2 + self.walker.BUFFER_LENGTH, parent_bone.width, parent_bone.width)
    child_frame_pos = Vec3(-self.walker.BUFFER_LENGTH, child_bone.width,child_bone.width)
    parent_frame = TransformState.makePos(parent_frame_pos)
    child_frame = TransformState.makePosHpr(child_frame_pos, Vec3(*hpr))
    constraint = BulletHingeConstraint(parent_bone.np.node(), child_bone.np.node(), parent_frame, child_frame)
    constraint.setLimit(-120, 120)
    constraint.enableFeedback(True)
    self.world.attachConstraint(constraint)
    return constraint

  def get_joint_angles(self):
    return [joint.constraint.getHingeAngle() for joint in self.walker.joints]

  def get_bones_height(self):
    return [bone.np.getPos()[2] for bone in self.walker.bones]

  def apply_action(self, action):
    for i in range(len(self.walker.joints)):
      self.walker.joints[i].constraint.enableAngularMotor(True, action[i], 3)


  def get_action(self, state):
    return 

  def get_score(self):
    com = self.get_com()
    com[2] = 0
    return com.length()

  def get_com(self):
    positions = [bone.np.getPos() for bone in self.walker.bones]
    com = Vec3(0,0,0)
    for p in positions:
      com += p
    com /= len(self.walker.bones)
    return com

  def spinCameraTask(self, task):
      angleDegrees = task.time * 6.0
      angleRadians = angleDegrees * (math.pi / 180.0)
      new_cam_pos = self.get_com() + Vec3(20 * math.sin(angleRadians), -20.0 * math.cos(angleRadians), 3)
      self.camera.setPos(new_cam_pos)
      self.camera.setHpr(angleDegrees, 0, 0)
      self.light_track()

      
      j = self.walker.joints[0]
      # for j in self.walker.joints:
        # import pdb; pdb.set_trace()
      return Task.cont

  def remove_shape(self):
    for joint in self.walker.joints:
      self.world.removeConstraint(joint.constraint)
    for bone in self.walker.bones:
      self.world.removeRigidBody(bone.np.node())
      bone.model.removeNode()
      bone.np.removeNode()
      
  def color_contacts(self):
    for bone in self.walker.bones:
      bone.np.setColor(0.6, 0.6, 1.0, 1.0)
    for bone in self.walker.bones:
      result = self.world.contactTest(bone.np.node())
      for contact in result.getContacts():
        np = self.render.find(contact.getNode0().getName())
        np.setColor(1.0, 0.6, 0.6, 1.0)
        # np = self.render.find(contact.getNode1().getName())
        # np.setColor(1.0, 0.6, 0.6, 1.0)
        # print(contact.getNode1().getName())

  def update_task(self, task):
    self.update_physics(task)
    self.color_contacts()
    return task.cont

  def step(self, action, dt):
    self.apply_action(action)
    self.world.doPhysics(dt)
    
  def update_physics(self, dummy):
    if self.dt > 0:
      dt = self.dt
    else:
      dt = globalClock.getDt()
    action = self.walker.gen_actions()
    self.step(action, dt)

  def runy(self, with_graphics = True, dt = 0):
    self.dt = dt
    taskMgr.add(self.update_task, 'update')
    taskMgr.add(self.spinCameraTask, "SpinCameraTask")
    if with_graphics:
      base.run()
    else:
      while True:
        self.update_physics(None)
