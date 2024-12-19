from importlib.resources import as_file, files
from PIL import Image
import numpy as np
import os
import sys
from contextlib import contextmanager

class Plane:
    def __init__(self, bullet_client):
        
        self._p = bullet_client
        self.client = self._p._client
        with as_file(files("bridges.resources").joinpath("bridge.urdf")) as f_name:
            with suppress_stdout():
                self.track = self._p.loadURDF(fileName=str(f_name), basePosition=[0, 0, 0])
        with as_file(files("bridges.resources").joinpath("track.urdf")) as f_name:
            with suppress_stdout():
                self.bridge = self._p.loadURDF(fileName=str(f_name), basePosition=[0, 0, 0])
        with as_file(files("bridges.resources").joinpath("wall1.urdf")) as f_name:
            with suppress_stdout():
                self.walls = self._p.loadURDF(fileName=str(f_name), basePosition=[0, 0, 0])

        # texture for wall
        with as_file(files("bridges.resources").joinpath("checkered5.png")) as texture_f_name:
            with suppress_stdout():
                self.wall_texture = self._p.loadTexture(str(texture_f_name))

        planeStartOrientation = self._p.getQuaternionFromEuler([0, 0, 0])
        wall1StartPos = [42.5,0,0]
        wall1 = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=[0.01, 63/2, 3])
        wallid1=self._p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,baseVisualShapeIndex=wall1, basePosition=wall1StartPos, baseOrientation=planeStartOrientation)
        self._p.changeVisualShape(wallid1, -1, textureUniqueId=self.wall_texture)

        with as_file(files("bridges.resources").joinpath("checkered.png")) as texture_swall1_f_name:
                self.side_wall1_texture = self._p.loadTexture(str(texture_swall1_f_name))
        planeStartOrientation = self._p.getQuaternionFromEuler([0, 0, 0])
        wall2StartPos = [0,-4,0]
        wall2 = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=[69/2+0.1, 0.01, 3])
        wallid2=self._p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,baseVisualShapeIndex=wall2, basePosition=wall2StartPos, baseOrientation=planeStartOrientation)
        self._p.changeVisualShape(wallid2, -1, textureUniqueId=self.side_wall1_texture)
        wall3StartPos = [0,4,0]
        wall3 = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=[69/2+0.1, 0.01, 3])
        wallid3=self._p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,baseVisualShapeIndex=wall3, basePosition=wall3StartPos, baseOrientation=planeStartOrientation)
        self._p.changeVisualShape(wallid3, -1, textureUniqueId=self.side_wall1_texture)

        bridge_axis = [0, 1, 0]  # z-axis
        desired_angle = -1.0993  # 60 degrees in radians
        quat = self._p.getQuaternionFromAxisAngle(bridge_axis, desired_angle)
        self._p.resetBasePositionAndOrientation(self.bridge, [0,0,0],quat)
        # self.lowerbridge()

    def lowerbridge(self):
            self._p.changeVisualShape(self.bridge, -1, rgbaColor=[0, 0, 0, 0])
            # bridge_axis = [0, 1, 0]  # z-axis
            # desired_angle = 0  # 60 degrees in radians
            # quat = self._p.getQuaternionFromAxisAngle(bridge_axis, desired_angle)
            # self._p.resetBasePositionAndOrientation(self.bridge, [0,0,0],quat)


# https://github.com/bulletphysics/bullet3/issues/2170
@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
