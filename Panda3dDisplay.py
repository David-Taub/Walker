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
        self.visualize_ground()
        for bone in self.physics.bones_to_nodes:
            self.visualize_bone(bone, self.physics.bones_to_nodes[bone])
        render.setAntialias(panda3d.core.AntialiasAttrib.MAuto)
        render.setShaderAuto()
        taskMgr.add(self.reposition_camera, "reposition_camera")
        taskMgr.add(self.reposition_light, "_reposition_light")
        logging.info('Visualization setup done')

    def reposition_camera(self, task):
        camera_offset = panda3d.core.Vec3(20, 20, 20)
        self.camera.setPos(self.physics.get_walker_position() + camera_offset)
        self.camera.lookAt(self.physics.get_walker_position())
        return Task.cont

    def reposition_light(self, task):
        light_offset = panda3d.core.Vec3(10, 30, 30)
        self.tracker_light_np.setPos(self.physics.get_walker_position() + light_offset)
        self.tracker_light_np.lookAt(self.physics.get_walker_position())
        return Task.cont

    # def create_camera(self):
    #     base.cam.setPos(0, 0, 0)
    #     base.cam.lookAt(0, 0, 0)
        # self._add_debug()

    def _add_debug(self):
        debug_node = panda3d.bullet.BulletDebugNode('Debug')
        debug_node.showWireframe(True)
        debug_node.showConstraints(True)
        debug_node.showBoundingBoxes(False)
        debug_node.showNormals(True)
        debug_np = render.attachNewNode(debug_node)
        debug_np.show()
        self.physics.world.setDebugNode(debug_np.node())

    def create_light(self):
        def create_ambient_light():
            ambient_light = panda3d.core.AmbientLight('ambientLight')
            ambient_light.setColor(panda3d.core.Vec4(0.5, 0.5, 0.5, 1))
            ambientLight_np = render.attachNewNode(ambient_light)
            render.setLight(ambientLight_np)

        # SpotLight
        def create_tracking_spotlight():
            self.tracker_light_np = render.attachNewNode(panda3d.core.Spotlight("tracker"))
            self.tracker_light_np.setPos(0, 50, 30)
            self.tracker_light_np.lookAt(0, 0, 0)
            self.tracker_light_np.node().setShadowCaster(True, 2048, 2048)
            self.tracker_light_np.node().getLens().setFov(90)
            render.setLight(self.tracker_light_np)

        def create_fixed_spotlight():
            fixed_light_np = render.attachNewNode(panda3d.core.Spotlight("Spot"))
            fixed_light_np.setPos(50, 0, 30)
            fixed_light_np.lookAt(0, 0, 0)
            fixed_light_np.node().setShadowCaster(True, 512, 512)
            fixed_light_np.node().getLens().setFov(170)
            render.setLight(fixed_light_np)

        create_ambient_light()
        create_tracking_spotlight()
        # create_fixed_spotlight()

    # def debug_screen_print(self, action, state, reward, score):
    #     try:
    #         self.textObject.destroy()
    #     except:
    #         pass

    #     text = "Score: %.2f Actions: %s, Reward: %.2f" % (score, [int(a) for a in action], reward)
    #     self.textObject = OnscreenText(text=text, pos=(0.1, 0.1), scale=0.07, align=TextNode.ALeft)
    #     self.textObject.reparentTo(base.a2dBottomLeft)

    def visualize_ground(self):
        card_maker = panda3d.core.CardMaker('')
        card_maker.setFrame(-100, 100, -100, 100)
        self.ground_node = card_maker.generate()
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

    def render_scene(self):
        taskMgr.step()
