import logging
import math

import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
from direct.showbase.ShowBase import ShowBase

import panda3d.core
import panda3d.bullet


class Panda3dDisplay(ShowBase):
    """
    responsible for the visual side of the scene:
    ground, bones boxes, light and camera
    """

    def __init__(self, physics):
        ShowBase.__init__(self)
        self.physics = physics
        self.font = loader.loadFont('courier_new_bold.ttf')
        self.create_light()
        self.show_debug_frames()
        self.visualize_ground()
        for bone in self.physics.bones_to_nodes:
            self.visualize_bone(bone, self.physics.bones_to_nodes[bone])
        render.setAntialias(panda3d.core.AntialiasAttrib.MAuto)
        render.setShaderAuto()
        taskMgr.add(self.reposition_camera, "reposition_camera")
        taskMgr.add(self.reposition_light, "_reposition_light")
        logging.debug('Visualization setup done')

    def close_window(self):
        base.destroy()

    def finalizeExit(self):
        pass

    def reposition_camera(self, task):
        camera_offset = panda3d.core.Vec3(30, 30, 30)
        self.camera.setPos(self.physics.get_walker_position() + camera_offset)
        self.camera.lookAt(self.physics.get_walker_position())
        return Task.cont

    def reposition_light(self, task):
        light_offset = panda3d.core.Vec3(0, 0, 100)
        self.tracker_light_np.setPos(self.physics.get_walker_position() + light_offset)
        self.tracker_light_np.lookAt(self.physics.get_walker_position())
        return Task.cont

    def show_debug_frames(self):
        debug_node = panda3d.bullet.BulletDebugNode('Debug')
        debug_node.drawMaskChanged()
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
            self.tracker_light_np.node().getLens().setFov(170)
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

    def debug_screen_print(self, text):
        try:
            self.textObject.destroy()
        except Exception:
            pass

        self.textObject = OnscreenText(text=text, pos=(0.1, 0.1 * text.count('\n')), scale=0.07,
                                       align=panda3d.core.TextNode.ALeft, fg=(1.0, 1.0, 1.0, 1.0),
                                       bg=(0.0, 0.0, 0.0, 0.5), font=self.font)
        self.textObject.reparentTo(base.a2dBottomLeft)

    def visualize_ground(self):
        card_maker = panda3d.core.CardMaker('')
        card_maker.setFrame(-100, 100, -100, 100)
        ground_np = render.attachNewNode(card_maker.generate())
        ground_np.lookAt(0, 0, -10)
        ground_np.setPos(panda3d.core.Vec3(0, 0, 0))
        ground_np.setTexture(loader.loadTexture('maps/grid.rgb'))
        card_maker.setFrame(100, 300, -100, 100)
        ground_np = render.attachNewNode(card_maker.generate())
        ground_np.lookAt(0, 0, -10)
        ground_np.setPos(panda3d.core.Vec3(0, 0, 0))
        ground_np.setTexture(loader.loadTexture('maps/grid.rgb'))

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
