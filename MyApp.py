from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
import math, random
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from panda3d.bullet import *


class MyApp(ShowBase):
  MASS = 1
  MOTOR_POWER = 1000
  def __init__(self, walker, is_displaying):
    # World
    self.is_displaying = is_displaying
    self.world = BulletWorld()
    self.world.setGravity(Vec3(0, 0, -9.81))
    self.walker = walker
    if self.is_displaying:
      self._init_display()


  def _init_display(self):
    print("Setting up display")
    ShowBase.__init__(self)
    self.add_light()
    taskMgr.add(self.spinCameraTask, "SpinCameraTask")
    base.cam.setPos(0, -50, 20)
    base.cam.lookAt(0, 0, 0)
    # self.add_debug()

  def add_debug(self):
    debug_node = BulletDebugNode('Debug')
    debug_node.showWireframe(True)
    debug_node.showConstraints(True)
    debug_node.showBoundingBoxes(False)
    debug_node.showNormals(True)
    debug_np = render.attachNewNode(debug_node)
    debug_np.show()
    self.world.setDebugNode(debug_np.node())

  def add_light(self):
    ambientLight = AmbientLight('ambientLight')
    ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
    ambientLightNP = render.attachNewNode(ambientLight)
    render.setLight(ambientLightNP)
    # SpotLight
    self.tracker_light_np = render.attachNewNode(Spotlight("tracker"))
    self.tracker_light_np.setPos(0,80,30)
    self.tracker_light_np.lookAt(0,0,0)
    self.tracker_light_np.node().setShadowCaster(True, 1024, 1024)
    self.tracker_light_np.node().getLens().setFov(90)
    render.setLight(self.tracker_light_np)
    slight = render.attachNewNode(Spotlight("Spot2"))
    slight.setPos(80,0,30)
    slight.lookAt(0,0,0)
    slight.node().setShadowCaster(True, 1024, 1024)
    slight.node().getLens().setFov(90)
    render.setLight(slight)
    render.setAntialias(AntialiasAttrib.MAuto)
    render.setShaderAuto()

  def light_track(self):
    self.tracker_light_np.setPos(self.get_com()+Vec3(0,80,30))
    self.tracker_light_np.lookAt(self.get_com())


  def init_plane(self):
    # Plane
    shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
    node = BulletRigidBodyNode('Ground')
    node.addShape(shape)
    node.setFriction(1)
    self.world.attachRigidBody(node)
    self.plane = Plane(Vec3(0, 0, 0), Point3(0, 0, 0))
    if self.is_displaying:
      cm = CardMaker('')
      cm.setFrame(-1000, 1000, -1000, 1000)
      self.ground_node = cm.generate()
      self.ground_np = render.attachNewNode(self.ground_node)
      self.ground_np.lookAt(0, 0, -1)
      self.ground_np.setTexture(loader.loadTexture('maps/grid.rgb'))


  def get_bones_positions(self):
    return [[bone.node.getTransform().getPos(), bone.node.getTransform().getHpr()] for bone in self.walker.bones]

  def init_bone(self, bone, position, index):
    # Box
    shape = BulletBoxShape(Vec3(bone.length, bone.height, bone.width))
    ts = TransformState.makePos(Point3(bone.length, bone.height, bone.width))
    bone.node = BulletRigidBodyNode('Bone%d' % index)
    bone.node.setMass(self.MASS)
    bone.node.setFriction(1)
    bone.node.addShape(shape, ts)
    bone.node.setTransform(TransformState.makePosHpr(position[0], position[1]))
    self.world.attachRigidBody(bone.node)

    if self.is_displaying:
      bone.np = render.attachNewNode(bone.node)
      # bone.np.setPos(position[0])
      # bone.np.setHpr(position[1])
      bone.np.setShaderAuto()
      bone.model = loader.loadModel('models/box.egg')
      bone.model.setScale(Vec3(2*bone.length, 2*bone.height, 2*bone.width))
      bone.model.reparentTo(bone.np)
      bone.np.setColor(0.6, 0.6, 1.0, 1.0)

  def add_joint_ball(self, bone):
    if bone.has_joint_ball:
      return
    model = loader.loadModel('smiley.egg')
    model.reparentTo(render)
    scale_vec = Vec3(self.walker.BUFFER_LENGTH,self.walker.BUFFER_LENGTH,self.walker.BUFFER_LENGTH)
    model.setTransform(TransformState.makePos(Point3(bone.length * 2 + self.walker.BUFFER_LENGTH, bone.height, bone.width)))
    model.setScale(scale_vec)
    tex = loader.loadTexture('maps/noise.rgb')
    model.setTexture(tex, 1)
    model.reparentTo(bone.np)

    bone.has_joint_ball = True

  def add_joint(self, parent_bone, child_bone, hpr, pos):
    # self.add_joint_ball(parent_bone)
    parent_frame_pos = Vec3(parent_bone.length * 2 + self.walker.BUFFER_LENGTH, parent_bone.height, parent_bone.width)
    child_frame_pos = Vec3(-self.walker.BUFFER_LENGTH, child_bone.height, child_bone.width)
    parent_frame = TransformState.makePos(parent_frame_pos)
    child_frame = TransformState.makePosHpr(child_frame_pos, Vec3(*hpr))
    constraint = BulletHingeConstraint(parent_bone.node, child_bone.node, parent_frame, child_frame)
    constraint.setLimit(-120, 120)
    constraint.enableFeedback(True)
    self.world.attachConstraint(constraint)
    return constraint

  def get_joint_angles(self):
    return [joint.constraint.getHingeAngle() for joint in self.walker.joints]

  def get_bones_z(self):
    return [bone.node.getTransform().getPos()[2] for bone in self.walker.bones]

  def apply_action(self, action):
    for i in range(len(self.walker.joints)):
      self.walker.joints[i].constraint.enableAngularMotor(True, action[i], self.MOTOR_POWER)


  def get_com(self):
    positions = [bone.node.getTransform().getPos() for bone in self.walker.bones]
    com = Vec3(0, 0, 0)
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
    # for bone in self.walker.bones:
    #   if self.is_displaying:
    #     bone.model.removeNode()
    #     bone.np.removeNode()

    if self.is_displaying:
      [np.removeNode() for np in render.getChildren() if np.getName().startswith("Bone")]


    for body in self.world.getRigidBodies():
      self.world.removeRigidBody(body)
    for constraint in self.world.getConstraints():
      self.world.removeConstraint(constraint)

    if self.is_displaying:
      self.ground_np.removeNode()
      self.ground_node.removeAllChildren()

  def color_contacts(self):
    self.walker.bones[0].np.setColor(1.0, 0.6, 0.6, 1.0)

  def physical_step(self, action, dt):
    self.apply_action(action)
    self.world.doPhysics(dt)
    if self.is_displaying:
      # render frame
      taskMgr.step()

  def head_hpr(self):
    return self.walker.bones[0].node.getTransform().getPos()

  def debug_screen_print(self, action, state, reward, score):
    if not self.is_displaying:
      return
    try:
      self.textObject.destroy()
    except:
      pass

    text = "Score: %.2f Actions: %s" % (score, [int(a) for a in action])
    self.textObject = OnscreenText(text = text, pos = (0.1, 0.1), scale = 0.07, align = TextNode.ALeft)
    self.textObject.reparentTo(base.a2dBottomLeft)