import math
from importlib.resources import as_file, files
import os
import sys
from contextlib import contextmanager

class mouse:
    def __init__(self,bullet_client,time):
        self._p = bullet_client
        self.client = self._p._client
        with as_file(files("bridges.resources").joinpath("mouse.urdf")) as f_name:
            with suppress_stdout():
                self.mouse = self._p.loadURDF(fileName=str(f_name), basePosition=[-69/2, 0, 2])
        self.anglediff=0
        self.time_int=time
        self.flag=False


    def get_ids(self):
        return self.mouse, self.client
    
    def get_coordinates(self):
        position= self._p.getBasePositionAndOrientation(self.mouse)[0]
        position = position[:2]
        return position
    
    def get_angle(self):
        angle= self._p.getBasePositionAndOrientation(self.mouse)[1]
        radians_ang = self._p.getEulerFromQuaternion(angle)[2]
        return math.degrees(radians_ang)

    def greenflag(self):
         self.flag=True

    def pointposition1(self,a,b):
        return (b+1.14*a-69.50)>0
    
    def pointposition2(self,a,b):
        return (b+1.08*a-75.58)<0

    def pointposition3(self,a,b):
        return (b-1.14*a-69.50)>0

    def pointposition4(self,a,b):
        return (b-1.08*a-75.58)<0

    def takeaction(self, action):

        crosswall="None"

        steering_angle = action

        #left
        if steering_angle==0:
            steering_angle=math.pi/12

        #right
        elif steering_angle==1:
            steering_angle=-math.pi/12
        #straight
        else:
            steering_angle=0

        #controls steps length
        speed=2

        # gets current position,angle
        pos,ori=self._p.getBasePositionAndOrientation(self.mouse)



        # print(ori)

        #new angle
        radians_ori = self._p.getEulerFromQuaternion(ori)[2]


        newpos=pos
        ang=radians_ori + steering_angle

        if newpos[0]<35:
            ang = radians_ori + steering_angle/2
            if ang>=math.pi/12:
                ang=math.pi/12-0.001
                crosswall="angle out of bounds"
            if ang<=-math.pi/12:
                ang=-math.pi/12+0.001
                crosswall="angle out of bounds"

        if steering_angle==0:

            xdif = pos[1] + speed*math.sin(ang)
            ydif = pos[0] + speed*math.cos(ang)

            if pos[0]<35:
                xdif = pos[1]
                ydif = pos[0] + speed

            newpos= [ydif,xdif,pos[2]]
            # maintains behavior in the vertical track before the red line
            if not self.flag:
                #makes sure that it doesn't crosses bridge and the wall bounds
                    if newpos[0] < -35:
                        newpos[0]=-35
                        crosswall="initial point"
                    if newpos[0]>-12:
                        newpos[0]=-12
                    if newpos[1]<=-3.95:
                        newpos[1]=-3.9
                        crosswall="Right wall on Vertical track"
                    if newpos[1]>=3.95:
                        newpos[1]=3.9
                        crosswall="Left wall on Vertical track"

            # maintains behavior in the track after the red line
            else:
                if pos[0]>35:
                    if newpos[0]<=36:
                        newpos[0]=36
                        crosswall="Back wall on Horizontal track"
                    elif newpos[0]>41.25:
                        newpos[0]=41.25
                        crosswall="Front wall on Horizontal track"
                #this is for vertical portion, which makes sure that the agent doesn't crosses the side walls
                # if newpos[0]<=35 and newpos[1]>-3.95 and newpos[1]<3.95:
                #     if newpos[1]<=-3.95:
                #         newpos[1]=-3.9
                #         crosswall="Right wall on Vertical track"
                #     if newpos[1]>=3.95:
                #         newpos[1]=3.9
                #         crosswall="Left wall on Vertical track"

                # #this is for horizontal portion, prevents agent to collide from back wall
                # elif newpos[0]<=35 and not (newpos[1]>-3.95 and newpos[1]<3.95):
                #     if newpos[0]<=35:
                #         newpos[0]=35
                #         crosswall="Back wall on Horizontal track"

                # #this is for horizontal portion
                # else:
                #     #this is for the front wall in the horizontal track
                #     if newpos[0]>42.35:
                #         newpos[0]=42.35
                #         crosswall="Front wall on Horizontal track"


        # print(ang)
        # if steering_angle==0:

        #     # before the red line
        #     if not self.flag:
        #         if ydif >= -35 and ydif<=-12 and xdif>-3.95 and xdif<3.95:
        #                 newpos= [ydif,xdif,pos[2]]
        #     else:
        #         if ydif >= -34.5 and ydif<=35 and xdif>-3.95 and xdif<3.95:
        #                 # newpos= [ydif,xdif,pos[2]]
        #                 newpos= [pos[0]+speed,pos[1],pos[2]]
        #         elif  ydif > 33 and ydif<=44 and xdif>-28 and xdif<28:
        #                 if ydif>42.35:
        #                      ydif=42.35
        #                 if ydif<35:
        #                      ydif=35
        #                 newpos= [ydif,xdif,pos[2]]
        #         elif self.pointposition1(xdif,ydif) and xdif<31.5 and ydif<=42.45:
        #                 newpos= [ydif,xdif,pos[2]]
        #         elif self.pointposition1(xdif,ydif) and self.pointposition2(xdif,ydif) and xdif>=31.5:
        #                 newpos= [ydif,xdif,pos[2]]

        #         elif self.pointposition3(xdif,ydif) and xdif>-31.5 and ydif<=42.45:
        #                 newpos= [ydif,xdif,pos[2]]
        #         elif self.pointposition3(xdif,ydif) and self.pointposition4(xdif,ydif) and xdif<=-31.5:
        #                 newpos= [ydif,xdif,pos[2]]
        # # print(xdif,ydif)

        # if steering_angle>0 or steering_angle<0:
        #     if  ydif > 33 and ydif<=44 and xdif>-28 and xdif<28:
        #         if ydif>42.35:
        #                 ydif=42.35
        #         if ydif<35:
        #                 ydif=35
        #         newpos= [ydif,xdif,pos[2]]


        rotation_axis = [0, 0, 1]  # z-axis
        rotation_angle = self._p.getQuaternionFromAxisAngle(rotation_axis, ang)

        # print("angle")
        # print(math.degrees(ang))

        # print("position")
        # print(newpos)

        self._p.resetBasePositionAndOrientation(self.mouse, newpos,rotation_angle)
        return crosswall


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





