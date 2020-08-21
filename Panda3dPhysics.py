import logging

import numpy as np

import panda3d.bullet
from panda3d.core import Vec3
from panda3d.core import Point3
from panda3d.core import TransformState

JOINT_POWER = 20
JOINT_SPEED = 0.75
PLANE_FRICTION = 1
GRAVITY_ACCELERATION = 9.81
STEP_TIME = 1


class Panda3dPhysics:
    def __init__(self):
        self.world = panda3d.bullet.BulletWorld()
        self.world.setGravity(Vec3(0, 0, -GRAVITY_ACCELERATION))
        self._create_ground()

    def _create_ground(self):
        # Plane
        logging.debug("Ground plane added to physics engine")
        plane_shape = panda3d.bullet.BulletPlaneShape(Vec3(0, 0, 10), 1)
        self.ground_node = panda3d.bullet.BulletRigidBodyNode('Ground')
        self.ground_node.addShape(plane_shape)
        self.ground_node.setTransform(TransformState.makePos(Vec3(0, 0, -1)))
        self.ground_node.setFriction(PLANE_FRICTION)
        self.world.attachRigidBody(self.ground_node)

    def add_walker(self, walker):
        logging.debug("adding walker to physics engine")
        # self.walker = Spider()
        # self.walker = Shape()
        self.bones_to_nodes = {}
        self.constraints = []
        [self._create_bone_node(bone) for bone in walker.bones]
        [self._create_joint_constraint(joint) for joint in walker.joints]

    def _create_bone_node(self, bone):
        box_shape = panda3d.bullet.BulletBoxShape(Vec3(bone.length, bone.height, bone.width))
        # ts = TransformState.makePos(Point3(bone.length, bone.height, bone.width))
        ts = TransformState.makePos(Point3(0, 0, 0))
        bone_node = panda3d.bullet.BulletRigidBodyNode(bone.name)
        bone_node.setMass(bone.mass)
        bone_node.setFriction(bone.friction)
        bone_node.addShape(box_shape, ts)
        bone_node.setTransform(TransformState.makePosHpr(Vec3(*bone.start_pos), Vec3(*bone.start_hpr)))
        self.world.attachRigidBody(bone_node)
        self.bones_to_nodes[bone] = bone_node

    def _create_joint_constraint(self, joint):
        parent_bone = joint.parent_bone
        child_bone = joint.child_bone
        parent_frame_pos = Vec3(parent_bone.length + joint.gap_radius, 0, 0)
        child_frame_pos = Vec3(-child_bone.length - joint.gap_radius, 0, 0)
        parent_frame = TransformState.makePos(parent_frame_pos)
        child_frame = TransformState.makePosHpr(child_frame_pos, Vec3(*joint.start_hpr))
        constraint = panda3d.bullet.BulletHingeConstraint(
            self.bones_to_nodes[parent_bone], self.bones_to_nodes[child_bone], parent_frame, child_frame)
        constraint.setLimit(*joint.angle_range)
        constraint.enableFeedback(False)
        self.world.attachConstraint(constraint)
        self.constraints.append(constraint)

    def get_bones_positions(self):
        return np.array([node.getTransform().getPos() for node in self._get_ordered_bone_nodes()])

    def get_bones_relative_positions(self):
        walker_position = self.get_walker_position()
        return np.array([node.getTransform().getPos() - walker_position for node in self._get_ordered_bone_nodes()])

    def get_bones_orientations(self):
        return np.array([node.getTransform().getHpr() for node in self._get_ordered_bone_nodes()])

    def get_bones_linear_velocity(self):
        return np.array([node.getLinearVelocity() for node in self._get_ordered_bone_nodes()])

    def get_bones_angular_velocity(self):
        return np.array([node.getAngularVelocity() for node in self._get_ordered_bone_nodes()])

    def set_bones_pos_hpr(self, positions, orientations):
        # position - n x 3 array
        for index, node in enumerate(self._get_ordered_bone_nodes()):
            transform = TransformState.makePosHpr(Vec3(*positions[index]), Vec3(*orientations[index]))
            node.setTransform(transform)
            node.setLinearVelocity(Vec3(0, 0, 0))
            node.setAngularVelocity(Vec3(0, 0, 0))

    def _get_ordered_bone_nodes(self):
        bones = list(self.bones_to_nodes.keys())
        bones.sort(key=lambda x: x.index)
        return [self.bones_to_nodes[bone] for bone in bones]
    # def _create_joint_constraints_ball(self, bone):
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

    def get_joint_angles(self):
        return np.array([constraint.getHingeAngle() for constraint in self.constraints])

    # def get_bones_z(self):
    #     return [node.getTransform().getPos()[2] for node in self.bones_to_nodes]

    def apply_action(self, action):
        if action is None:
            action = np.zeros([len(self.constraints)])
        for index in range(len(self.constraints)):
            self.constraints[index].enableAngularMotor(True, action[index] * JOINT_SPEED, JOINT_POWER)

    def get_walker_position(self):
        return sum([bone_node.getTransform().getPos() for bone_node in self.bones_to_nodes.values()],
                   Vec3(0, 0, 0)) / len(self.bones_to_nodes)
        # bone_node = list(self.bones_to_nodes.values())[0]
        # return bone_node.getTransform().getPos()

    def step(self):
        self.world.doPhysics(STEP_TIME)

    # def remove_shape(self):

    #     if self.is_displaying:
    #         [np.removeNode() for np in render.getChildren() if np.getName().startswith("Bone")]

    #     for body in self.world.getRigidBodies():
    #         self.world.removeRigidBody(body)
    #     for constraint in self.world.getConstraints():
    #         self.world.removeConstraint(constraint)

    #     if self.is_displaying:
    #         self.ground_np.removeNode()
    #         self.ground_node.removeAllChildren()

    # def get_contacts(self):
    #     result = self.world.contactTest(self.ground_node)
    #     names = [contact.getNode0().getName() for contact in result.getContacts()]
    #     indices = [int(name[4]) for name in names if name.startswith("Bone")]
    #     contacts = np.zeros(len(self.bones_to_nodes))
    #     contacts[indices] = 1
    #     return contacts
