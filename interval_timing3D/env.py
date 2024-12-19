import gymnasium as gym
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
from bridges.resources.mouse import mouse
from bridges.resources.plane import Plane
import matplotlib.pyplot as plt
import random
import math


class TempBisectionEnv3D(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, time_intervals, height_obs, width_obs, dt, reward_for_steps, reward_abort, reward_incorrect, reward_correct, render_mode):

        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=(height_obs, width_obs, 3))

        self._p = bullet_client.BulletClient(connection_mode=None)
        self.client = self._p._client
        self._p.setTimeStep(1 / 30)

        self.mouse = None
        self.plane = None
        self.goal = None
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self.startsteps = None
        self.totalsteps = None
        self.elapsed_time = 0
        self.time_intervals = time_intervals
        self.height_obs = height_obs
        self.width_obs = width_obs
        self.dt = dt
        self.reward_for_steps = reward_for_steps
        self.reward_abort = reward_abort
        self.reward_incorrect = reward_incorrect
        self.reward_correct = reward_correct
        self.overallactions=0
        self.overalltotalactions=None
        self.render_mode = render_mode
        self.decided=None
        self.flag = False
        self.reset()

    def step(self, action):

        self.overallactions+=1
        wallcollision=self.mouse.takeaction(action)
        self._p.stepSimulation()
        mouse_pos = self.mouse.get_coordinates()

        reward = self.reward_for_steps

        evidencedict = {
            "position": [mouse_pos[0], mouse_pos[1]],
            "decision": None,
            "goal": self.goal,
            "time": self.time_interval,
            "time_interval": None,
            "reason_to_terminate": None,
            "angle": None,
        }

        mouse_ang = self.mouse.get_angle()
        #print(mouse_ang)
        evidencedict["angle"] = self.mouse.get_angle()
        # This condition initiates the variables for the time duration after the agent crosses the red line, and waits for bridge to go down
        if mouse_pos[0] > -14:
            if self.startsteps is None:
                evidencedict["time_interval"] = "started"
                self.startsteps = 1

        # This condition gives signal to environment, for lowering the bridge and allowing agent to go forward
        if self.startsteps is not None and self.flag is False:
            self.startsteps += 1
            if self.startsteps > self.totalsteps:
                self.plane.lowerbridge()
                self.mouse.greenflag()
                evidencedict["time_interval"] = "finished"
                self.flag = True

        # This condition sets a flag for environment to prevent it to make a complete turn
        if mouse_pos[0]>34 and (self.decided is None) and (mouse_ang>15 or mouse_ang<-15):
            if mouse_ang>15:
                self.decided=90
            elif mouse_ang<-15:
                self.decided=-90


        # This condition tells if the agent has reached the end (to the right/wrong goal or not)
        # if mouse_pos[0]<24 and (mouse_pos[1] < -34.5 or mouse_pos[1] > 34.5):
        if (mouse_pos[1] < -10 or mouse_pos[1] >10):
            self.done = True

            if (self.goal == "right" and mouse_pos[1] < 0) or (self.goal == "left" and mouse_pos[1] > 0):
                evidencedict['decision'] = True
                reward = self.reward_correct
            else:
                evidencedict['decision'] = False
                reward = self.reward_incorrect

            evidencedict["reason_to_terminate"] = "Reached goal"
            # print(evidencedict)

        # This condition prevents the agent to go back to the vertical track, in this case the environment terminates
        elif (mouse_ang < -135 or mouse_ang > 135) and (mouse_pos[1] > -4 and mouse_pos[1] < 4) and (mouse_pos[0] > -36 and mouse_pos[0] <= 35):
            self.done = True
            reward = self.reward_abort
            evidencedict['decision'] = False
            evidencedict["reason_to_terminate"] = "Going back on the track"

        # This condition prevents the agent to go beyond the walls, in this case the environement terminates
        if wallcollision is not "None" and not self.done:
            # self.done = True
            reward = self.reward_abort
            # evidencedict['decision'] = False
            if wallcollision=="angle out of bounds":
                evidencedict["reason_to_terminate"] = "Angle bounds on vertical track : " + wallcollision
            else:
                evidencedict["reason_to_terminate"] = "Tried to go beyond the walls : " + wallcollision

        # This condition prevents environment to go beyond required actions, ensuring the environment efficiency
        if self.overallactions>self.overalltotalactions and not self.done:
            self.done=True
            reward = self.reward_abort
            evidencedict['decision'] = False
            evidencedict["reason_to_terminate"] = "Overall total actions reached"

        #ob = np.array(mouse_pos, dtype=np.float32)
        # print(ob)
        ob = self.render()
        if evidencedict["time_interval"] == "started":
            # Define the white color range
            upper_white = np.array([255, 255, 255])
            # Create a binary mask for fully white pixels
            white_mask = np.all(ob == upper_white, axis=-1)
            # Set the fully white pixels to black
            ob[white_mask] = [0, 0, 0]

        trunc = False
        return ob, reward, self.done, trunc, evidencedict

    def reset(self, *, seed=None, options=None):

        self.time_interval = random.choice(self.time_intervals)
        self.startsteps = None
        self.totalsteps = int(math.ceil(self.time_interval/self.dt))
        self.elapsed_time = 0
        self.overallactions=0
        self.flag = False
        # on an average it takes around 130 steps to reach the end.
        self.overalltotalactions = 210 + self.totalsteps

        self.decided=None

        #self.time_interval = 4784
        #print("time interval:", self.time_interval)
        if self.time_interval > self.time_intervals[(len(self.time_intervals)//2)-1]:
            self.goal = "right"
        else:
            self.goal = "left"

        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setGravity(0, 0, 0)
        # Reload the plane and mouse
        self.plane = Plane(self._p)
        self.mouse = mouse(self._p, self.time_interval)
        self.done = False
        self.overallactions=0
        mouse_ob = self.render()
        evidencedict = {
            "decision": None,
            "goal": self.goal,
            "time": self.time_interval,
            "time_interval": "not_started",
            "reason_to_terminate":None
        }

        return mouse_ob, evidencedict

    def render(self):
        # mode = 'rgb_array' will not render the frames using matplotlib
        aspect_ratio = self.width_obs / self.height_obs
        mouse_id, _ = self.mouse.get_ids()
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=100, aspect=aspect_ratio, nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in self._p.getBasePositionAndOrientation(mouse_id)]
        rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        frame = self._p.getCameraImage(self.width_obs, self.height_obs, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        frame = np.array(frame, dtype=np.uint8)
        frame = np.reshape(frame, (self.height_obs, self.width_obs, 4))  # reshape to RBGA
        frame = frame[:, :, :3]  # RGB
        if self.render_mode == "rgb_array":
            return frame


    def close(self):
        self._p.disconnect()

class ScaleObs(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_space.shape[0], self.observation_space.shape[1], 1), dtype=float)

    def observation(self, observation):
        # grayscale conversion, Gray = 0.299 * R + 0.587 * G + 0.114 * B
        observation = observation @ np.array([[0.299], [0.587], [0.114]], dtype=float)
        observation /= 255.     # normalize

        return observation

def make_env3D(env_config):
    env = TempBisectionEnv3D(**env_config)
    env = ScaleObs(env)
    return env
