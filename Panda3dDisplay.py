import logging
import math

import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
from direct.showbase.ShowBase import ShowBase

import panda3d.core
import panda3d.bullet


PLANE_FRICTION = 100
GRAVITY_ACCELERATION = 9.81


class Panda3dDisplay(ShowBase):
    def _init_display(self):

        logging.debug("Setting up display")
        ShowBase.__init__(self)
        self._add_light()
        taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        base.cam.setPos(0, -40, 20)
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

    def _light_track(self):
        self.tracker_light_np.setPos(self.get_center_of_mass() + Vec3(0, 80, 30))
        self.tracker_light_np.lookAt(self.get_center_of_mass())

    def _add_light(self):
        ambientLight = panda3d.core.AmbientLight('ambientLight')
        ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        ambientLightNP = render.attachNewNode(ambientLight)
        render.setLight(ambientLightNP)
        # SpotLight
        self.tracker_light_np = render.attachNewNode(Spotlight("tracker"))
        self.tracker_light_np.setPos(0, 80, 30)
        self.tracker_light_np.lookAt(0, 0, 0)
        self.tracker_light_np.node().setShadowCaster(True, 1024, 1024)
        self.tracker_light_np.node().getLens().setFov(90)
        render.setLight(self.tracker_light_np)
        slight = render.attachNewNode(Spotlight("Spot2"))
        slight.setPos(80, 0, 30)
        slight.lookAt(0, 0, 0)
        slight.node().setShadowCaster(True, 1024, 1024)
        slight.node().getLens().setFov(90)
        render.setLight(slight)
        render.setAntialias(AntialiasAttrib.MAuto)
        render.setShaderAuto()

    def spinCameraTask(self, task):
        CAMERA_DISTANCE = 8
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (math.pi / 180.0)
        new_cam_pos = self.get_center_of_mass() + (Vec3(math.sin(angleRadians), -math.cos(angleRadians), 0.05) * CAMERA_DISTANCE)
        self.camera.setPos(new_cam_pos)
        self.camera.lookAt(self.get_center_of_mass())
        self._light_track()
        return Task.cont

    def debug_screen_print(self, action, state, reward, score):
        try:
            self.textObject.destroy()
        except:
            pass

        text = "Score: %.2f Actions: %s, Reward: %.2f" % (score, [int(a) for a in action], reward)
        self.textObject = OnscreenText(text=text, pos=(0.1, 0.1), scale=0.07, align=TextNode.ALeft)
        self.textObject.reparentTo(base.a2dBottomLeft)

    def add_textures(self, ):
        cm = CardMaker('')
        cm.setFrame(-100, 100, -100, 100)
        self.ground_node = cm.generate()
        self.ground_np = render.attachNewNode(self.ground_node)
        self.ground_np.lookAt(0, 0, -10)
        self.ground_np.setPos(Vec3(0, 0, 0))
        self.ground_np.setTexture(loader.loadTexture('maps/grid.rgb'))

    def visualize_bone(self, bone, bone_node):
        np = render.attachNewNode(bone_node)
        np.setShaderAuto()
        bone.model = loader.loadModel('models/box.egg')
        bone.model.setScale(Vec3(bone.length, bone.height, bone.width) * 2)
        bone.model.setPos(Vec3(-bone.length, -bone.height, -bone.width))
        bone.model.reparentTo(np)
        np.setColor(0.6, 0.6, 1.0, 1.0)

    def render2(self):
        taskMgr.step()
