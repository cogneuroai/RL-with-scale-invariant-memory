from turtle import width
import gym
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
from bridges.resources.mouse import mouse
from bridges.resources.plane import Plane
import matplotlib.pyplot as plt
import random
from bridges.envs.configure import time_interval,width_obs,height_obs,reward_for_steps,reward_incorrect,reward_correct,d_time
import time
import math



class TempBisectionEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode):
        self.render_mode = render_mode


        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=(height_obs, width_obs, 3))


        self._p = bullet_client.BulletClient(connection_mode=None)
        self.client = self._p._client
        self._p.setTimeStep(1/30)

        self.mouse = None
        self.plane = None
        self.goal = None
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self.start_time=None
        self.elapsed_time=0
        self.time_interval=time_interval
        self.startsteps = None
        self.totalsteps = 0
        self.flag=False
        self.reset()

    def step(self, action):

        self.mouse.takeaction(action)
        self._p.stepSimulation()
        mouse_pos = self.mouse.get_coordinates()
        
        reward=reward_for_steps

        evidencedict={
            "position": [mouse_pos[0],mouse_pos[1]],
            "decision":None,
            "Goal":self.goal,
            "time":self.time_interval,
            "Time Interval":"Not Started"
        }

        mouse_ang = self.mouse.get_angle()
        print(mouse_ang)
        if mouse_pos[0]>-14:
            # if self.start_time is None:
            #     self.start_time=time.time()*1000
            if self.startsteps is None:
                evidencedict["Time Interval"]="Started"
                self.startsteps=1

        if self.startsteps is not None and self.flag is False:
            # self.elapsed_time=time.time()*1000
            # if self.elapsed_time-self.start_time>self.time_interval:
            self.startsteps+=1
            if self.startsteps>self.totalsteps:
                self.plane.lowerbridge()
                self.mouse.greenflag()
                evidencedict["Time Interval"]="Finished",
                self.flag=True



        # if mouse_pos[0]<24 and (mouse_pos[1] < -34.5 or mouse_pos[1] > 34.5):
        if (mouse_pos[1] < -27 or mouse_pos[1] > 27):
            self.done = True

            if (self.goal == "right" and mouse_pos[1]<0) or (self.goal == "left" and mouse_pos[1]>0):
                evidencedict['decision']=True
                reward = reward_correct
            else:
                evidencedict['decision']=False
                reward = reward_incorrect
            # print(evidencedict)

        elif (mouse_ang < -135 or mouse_ang > 135) and (mouse_pos[1]>-4 and mouse_pos[1]<4 ) and (mouse_pos[0]>-34 and mouse_pos[0]<34):
            self.done = True
            reward = reward_incorrect
            evidencedict['decision']=False

        if mouse_pos[0]>=42.35:
                self.done=True
                reward=0
                evidencedict['decision']=False

        ob = np.array(mouse_pos, dtype=np.float32)
        # print(ob)
        ob= self.render()

        if self.render_mode == "human":
            plt.cla()
            plt.imshow(ob)
            plt.pause(0.0001)

        trunc = False
        return ob, reward, self.done, evidencedict

    def reset(self, *, seed=None, options=None):


        self.time_interval=random.choice(time_interval)
        self.totalsteps = int (math.ceil(self.time_interval/d_time))
        # self.time_interval=4784
        print("time interval:", self.time_interval)
        if self.time_interval>3700:
            self.goal="right"
        else:  
            self.goal = "left"
        
        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setGravity(0, 0, 0)
        # Reload the plane and mouse
        self.plane=Plane(self._p)
        self.mouse = mouse(self._p, self.time_interval)
        self.done = False
        mouse_ob = self.render()
        return mouse_ob, {}
        self.startsteps=None

    def render(self):
        aspect_ratio=width_obs/height_obs
        mouse_id, _ = self.mouse.get_ids()
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=150, aspect=aspect_ratio,
                                                nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    self._p.getBasePositionAndOrientation(mouse_id)]
        rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        frame = self._p.getCameraImage(width_obs,height_obs, view_matrix, proj_matrix,
                                       renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        frame = np.array(frame, dtype=np.uint8)
        frame = np.reshape(frame, (height_obs, width_obs, 4))  # reshape to RBGA
        frame = frame[:, :, :3]  # RGB

        return frame
 
        

    def close(self):
        self._p.disconnect()


def make_env3D(env_config):
    env = TempBisectionEnv(**env_config)
    return env
