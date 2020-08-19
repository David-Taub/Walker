import logging
import math

import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
from direct.showbase.ShowBase import ShowBase

import panda3d.core
import panda3d.bullet


class Panda3dDisplay(ShowBase):
    def __init__(self, physics):
        ShowBase.__init__(self)
        self.physics = physics
        self.create_light()
        self.create_camera()
        self.create_scene_visualization()
        logging.info('Visualization setup done')

    def create_camera(self):
        # taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        base.cam.setPos(0, -40, 20)
        base.cam.lookAt(0, 0, 0)
        # self._add_debug()

    # def _add_debug(self):
    #     debug_node = BulletDebugNode('Debug')
    #     debug_node.showWireframe(True)
    #     debug_node.showConstraints(True)
    #     debug_node.showBoundingBoxes(False)
    #     debug_node.showNormals(True)
    #     debug_np = render.attachNewNode(debug_node)
    #     debug_np.show()
    #     self.world.setDebugNode(debug_np.node())

    def _light_track(self):
        self.tracker_light_np.setPos(panda3d.core.Vec3(
            self.physics.get_walker_position()) + panda3d.core.Vec3(0, 80, 30))
        self.tracker_light_np.lookAt(panda3d.core.Vec3(self.physics.get_walker_position()))

    def create_light(self):
        ambientLight = panda3d.core.AmbientLight('ambientLight')
        ambientLight.setColor(panda3d.core.Vec4(0.2, 0.2, 0.2, 1))
        ambientLightNP = render.attachNewNode(ambientLight)
        render.setLight(ambientLightNP)
        # SpotLight
        self.tracker_light_np = render.attachNewNode(panda3d.core.Spotlight("tracker"))
        self.tracker_light_np.setPos(0, 80, 30)
        self.tracker_light_np.lookAt(0, 0, 0)
        self.tracker_light_np.node().setShadowCaster(True, 1024, 1024)
        self.tracker_light_np.node().getLens().setFov(90)
        render.setLight(self.tracker_light_np)
        slight = render.attachNewNode(panda3d.core.Spotlight("Spot2"))
        slight.setPos(80, 0, 30)
        slight.lookAt(0, 0, 0)
        slight.node().setShadowCaster(True, 1024, 1024)
        slight.node().getLens().setFov(90)
        render.setLight(slight)
        render.setAntialias(panda3d.core.AntialiasAttrib.MAuto)
        render.setShaderAuto()

    def spinCameraTask(self, task):
        CAMERA_DISTANCE = 8
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (math.pi / 180.0)
        new_cam_pos = self.physics.get_walker_position() + (
            panda3d.core.Vec3(math.sin(angleRadians),
                              - math.cos(angleRadians), 0.05) * CAMERA_DISTANCE)
        # self.camera.setPos(panda3d.core.Vec3(*new_cam_pos))
        # self.camera.lookAt(self.physics.get_walker_position())
        # self._light_track()
        return Task.cont

    # def debug_screen_print(self, action, state, reward, score):
    #     try:
    #         self.textObject.destroy()
    #     except:
    #         pass

    #     text = "Score: %.2f Actions: %s, Reward: %.2f" % (score, [int(a) for a in action], reward)
    #     self.textObject = OnscreenText(text=text, pos=(0.1, 0.1), scale=0.07, align=TextNode.ALeft)
    #     self.textObject.reparentTo(base.a2dBottomLeft)

    def create_scene_visualization(self):
        self.visualize_ground()
        for bone in self.physics.bones_to_nodes:
            self.visualize_bone(bone, self.physics.bones_to_nodes[bone])

    def visualize_ground(self):
        cm = panda3d.core.CardMaker('')
        cm.setFrame(-100, 100, -100, 100)
        self.ground_node = cm.generate()
        self.ground_np = render.attachNewNode(self.ground_node)
        self.ground_np.lookAt(0, 0, -10)
        self.ground_np.setPos(panda3d.core.Vec3(0, 0, 0))
        self.ground_np.setTexture(loader.loadTexture('maps/grid.rgb'))
        # self.ground_np.setColor(0.7, 0.1, 0.1, 1.0)

    def visualize_bone(self, bone, bone_node):
        bone_display_node = render.attachNewNode(bone_node)
        bone_display_node.setShaderAuto()
        bone.model = loader.loadModel('models/box.egg')
        bone.model.setScale(panda3d.core.Vec3(bone.length, bone.height, bone.width) * 2)
        bone.model.setPos(panda3d.core.Vec3(-bone.length, -bone.height, -bone.width))
        bone.model.reparentTo(bone_display_node)
        bone_display_node.setColor(0.6, 0.6, 1.0, 1.0)

    def render2(self):
        taskMgr.step()
