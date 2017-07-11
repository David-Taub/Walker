from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
import math, random
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from panda3d.bullet import *
import numpy as np

class Panda3dApp(ShowBase):
    def __init__(self, walker, is_displaying):
        # World
        self.is_displaying = is_displaying
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.walker = walker
        if self.is_displaying:
            self._init_display()
        self._init_plane()

    def load_shape(self, shape):
        self.shape = shape
        self.bone_nodes = []
        self.joint_constraints = []
        [self._init_bone(bone) for bone in self.shape.bones]
        [self._init_joint(joint) for joint in self.shape.joints]
        self.restart_bones_position()

    def _init_display(self):
        print("Setting up display")
        ShowBase.__init__(self)
        self._add_light()
        taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        base.cam.setPos(0, -50, 20)
        base.cam.lookAt(0, 0, 0)
        # self._add_debug()

    def _add_debug(self):
        debug_node = BulletDebugNode('Debug')
        debug_node.showWireframe(True)
        debug_node.showConstraints(True)
        debug_node.showBoundingBoxes(False)
        debug_node.showNormals(True)
        debug_np = render.attachNewNode(debug_node)
        debug_np.show()
        self.world.setDebugNode(debug_np.node())

    def _add_light(self):
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

    def _light_track(self):
        self.tracker_light_np.setPos(self.get_com()+Vec3(0,80,30))
        self.tracker_light_np.lookAt(self.get_com())


    def _init_plane(self):
        # Plane
        shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
        self.ground_node2 = BulletRigidBodyNode('Ground')
        self.ground_node2.addShape(shape)
        self.ground_node2.setFriction(1)
        self.world.attachRigidBody(self.ground_node2)
        self.plane = Plane(Vec3(0, 0, 0), Point3(0, 0, 0))
        if self.is_displaying:
            cm = CardMaker('')
            cm.setFrame(-1000, 1000, -1000, 1000)
            self.ground_node = cm.generate()
            self.ground_np = render.attachNewNode(self.ground_node)
            self.ground_np.lookAt(0, 0, -1)
            self.ground_np.setPos(Vec3(0,0,1))
            self.ground_np.setTexture(loader.loadTexture('maps/grid.rgb'))


    def save_shape_posture(self):
        com = self.get_com()
        com[2] = 0
        for bone in self.shape.bones:
            pos = self.bone_nodes[bone.index].getTransform().getPos()
            hpr = self.bone_nodes[bone.index].getTransform().getHpr()
            pos = pos - com
            bone.start_pos = Vec3(pos)
            bone.start_hpr = Vec3(hpr)

    def restart_bones_position(self):
        for bone in self.shape.bones:
            self.bone_nodes[bone.index].setTransform(TransformState.makePosHpr(Vec3(bone.start_pos),Vec3(bone.start_hpr)))

    def get_bones_positions(self):
        return [[list(node.getTransform().getPos()), list(node.getTransform().getHpr())] for node in self.bone_nodes]

    def _init_bone(self, bone):
        box_shape = BulletBoxShape(Vec3(bone.length, bone.height, bone.width))
        ts = TransformState.makePos(Point3(bone.length, bone.height, bone.width))
        node = BulletRigidBodyNode('Bone%d' % bone.index)
        self.bone_nodes.append(node)
        node.setMass(bone.mass)
        node.setFriction(bone.friction)
        node.addShape(box_shape, ts)
        node.setTransform(TransformState.makePosHpr(Vec3(*bone.start_pos), Vec3(*bone.start_hpr)))
        self.world.attachRigidBody(node)

        if self.is_displaying:
            np = render.attachNewNode(node)
            np.setShaderAuto()
            bone.model = loader.loadModel('models/box.egg')
            bone.model.setScale(Vec3(2*bone.length, 2*bone.height, 2*bone.width))
            bone.model.reparentTo(np)
            np.setColor(0.6, 0.6, 1.0, 1.0)

    # def _init_joint_ball(self, bone):
    #     if bone.has_joint_ball:
    #         return
    #     model = loader.loadModel('smiley.egg')
    #     model.reparentTo(render)
    #     scale_vec = Vec3(joint.gap_radius,joint.gap_radius,joint.gap_radius)
    #     model.setTransform(TransformState.makePos(Point3(bone.length * 2 + joint.gap_radius, bone.height, bone.width)))
    #     model.setScale(scale_vec)
    #     tex = loader.loadTexture('maps/noise.rgb')
    #     model.setTexture(tex, 1)
    #     model.reparentTo(bone.np)

    #     bone.has_joint_ball = True

    def _init_joint(self, joint):
        parent_bone = joint.parent_bone
        child_bone = joint.child_bone
        # self._init_joint_ball(parent_bone)
        parent_frame_pos = Vec3(parent_bone.length * 2 + joint.gap_radius, parent_bone.height, parent_bone.width)
        child_frame_pos = Vec3(-joint.gap_radius, child_bone.height, child_bone.width)
        parent_frame = TransformState.makePos(parent_frame_pos)
        child_frame = TransformState.makePosHpr(child_frame_pos, Vec3(joint.heading, joint.pitch, joint.roll))
        constraint = BulletHingeConstraint(self.bone_nodes[parent_bone.index], self.bone_nodes[child_bone.index], parent_frame, child_frame)
        constraint.setLimit(joint.min_angle, joint.max_angle)
        constraint.enableFeedback(False)
        self.world.attachConstraint(constraint)
        self.joint_constraints.append(constraint)

    def get_joint_angles(self):
        return [constraint.getHingeAngle() for constraint in self.joint_constraints]

    def get_bones_z(self):
        return [node.getTransform().getPos()[2] for node in self.bone_nodes]

    def apply_action(self, action):
        for i in range(len(self.shape.joints)):
            self.joint_constraints[i].enableAngularMotor(True, action[i]*self.shape.joints[i].action_factor, self.shape.joints[i].power)


    def get_com(self):
        positions = [node.getTransform().getPos() for node in self.bone_nodes]
        com = Vec3(0, 0, 0)
        for p in positions:
            com += p
        com /= len(self.bone_nodes)
        return com

    def spinCameraTask(self, task):
            angleDegrees = task.time * 6.0
            angleRadians = angleDegrees * (math.pi / 180.0)
            new_cam_pos = self.get_com() + Vec3(20 * math.sin(angleRadians), -20.0 * math.cos(angleRadians), 3)
            self.camera.setPos(new_cam_pos)
            self.camera.setHpr(angleDegrees, 0, 0)
            self._light_track()
            j = self.shape.joints[0]
            # for j in self.shape.joints:
                # import pdb; pdb.set_trace()
            return Task.cont

    def remove_shape(self):

        if self.is_displaying:
            [np.removeNode() for np in render.getChildren() if np.getName().startswith("Bone")]


        for body in self.world.getRigidBodies():
            self.world.removeRigidBody(body)
        for constraint in self.world.getConstraints():
            self.world.removeConstraint(constraint)

        if self.is_displaying:
            self.ground_np.removeNode()
            self.ground_node.removeAllChildren()

    def get_contacts(self):
        result = self.world.contactTest(self.ground_node2)
        names = [contact.getNode0().getName() for contact in result.getContacts()]
        indices = [int(name[4]) for name in names if name.startswith("Bone")]
        contacts = np.zeros(len(self.shape.bones))
        contacts[indices] = 1
        return contacts

    def step(self, action, dt, physical_steps_in_step):
        for i in range(physical_steps_in_step):
            self.apply_action(action)
            self.world.doPhysics(dt/physical_steps_in_step)
            if self.is_displaying:
                taskMgr.step()

    def head_hpr(self):
        return self.shape.bones[0].node.getTransform().getPos()

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