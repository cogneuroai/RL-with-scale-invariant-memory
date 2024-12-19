# simple gymnasium env of interval timing/temporal bisection for RL agents
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType


# create the class inheriting from gymnasium.env
# class should have methods {reset, step, render, close}
# construction should take in base intervals (a list), scale (to scale the intervals), \
# timings {fixation, stimulus, and decision periods}, rewards {correct/incorrect/abort endings}
# action dim-> Discrete(3) {fixation/stay/wait, choice1/turn-left, choice2/turn-right}
# reset->randomly select the interval, get gt, return info on trial
# step->return observation (according to the period the trial is in), reward (according to the action taken),\
# termination, truncation, info (which period it is in)


class IntervalTiming1D(gym.Env):

    metadata = {}

    def __init__(
            self,
            height,
            width,
            stimuli: list[int],
            scale: int = 1,
            timings: dict[str, Any] | None = None,
            rewards: dict[str, Any] | None = None,
            seed_offset: int = 0,
    ):
        self.height = height
        self.width = width
        self.stimuli = stimuli
        self.scale = scale
        self.timings = timings
        self.rewards = rewards
        self.seed_offset = seed_offset

        self.observation_space = gym.spaces.Box(0., 1., shape=(self.height, ), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

        #self.timings['fixation'] = self.timings['fixation'] * self.scale
        #self.timings['delay'] = self.timings['delay'] * self.scale
        #self.timings['decision'] = self.timings['decision'] * self.scale
        self.timings['fixation'] = self.timings['fixation']
        self.timings['delay'] = self.timings['delay']
        self.timings['decision'] = self.timings['decision']
        self.fixation_obs = [0.] * self.timings['fixation']
        self.stim_obs = []
        self.delay_obs = [0.] * self.timings['delay']
        self.decision_obs = [0.] * self.timings['decision']
        self.sampled_interval = None
        self.trial_obs = None
        self.t = 0  # trial timestep
        self.info = None
        self.gt = None

    def in_period(self, name: str):
        if 0 < self.t <= self.timings['fixation']:
            period = 'fixation'
        elif self.timings['fixation'] < self.t <= (self.timings['fixation']+self.timings['stimulus']):
            period = 'stimulus'
        elif (self.timings['fixation']+self.timings['stimulus']) < self.t <= (self.timings['fixation']+self.timings['stimulus']+self.timings['delay']):
            period = 'delay'
        else:
            period = 'decision'

        if name == period:
            return True
        else:
            return False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed+self.seed_offset)
        #self.sampled_interval = self.stimuli[self.np_random.integers(0, len(self.stimuli), size=1, dtype=int)]
        self.sampled_interval = self.np_random.choice(self.stimuli)   # randomly choose a single value from self.stimuli
        self.timings['stimulus'] = self.sampled_interval*self.scale
        # form the trial observations
        #self.stim_obs = [1.]*self.scale+[0.]*(self.timings['stimulus']-2*self.scale)+[1.]*self.scale
        self.stim_obs = [1.] + [0.] * (self.timings['stimulus'] - 2) + [1.]
        self.trial_obs = self.fixation_obs + self.stim_obs + self.delay_obs + self.decision_obs

        self.t = 0
        self.gt = 1 if self.sampled_interval >= self.stimuli[len(self.stimuli)//2] else 0
        self.info = {
            'trial_interval': self.sampled_interval,
            'period': 'fixation',
            'gt': self.gt,
            'current_t': self.t
            }
        return np.array(self.trial_obs[self.t]).reshape(-1), self.info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        self.t += 1

        if self.in_period('fixation'):
            self.info['period'] = 'fixation'
        elif self.in_period('stimulus'):
            self.info['period'] = 'stimulus'
        elif self.in_period('delay'):
            self.info['period'] = 'delay'
        elif self.in_period('decision'):
            if action == self.gt:
                reward = self.rewards['correct']
            else:
                reward = self.rewards['incorrect']
            done = True
            self.info['period'] = 'decision'
            self.info['current_t'] = self.t
            return np.array(self.trial_obs[self.t]).reshape(-1), reward, done, False, self.info

        reward = 0.
        done = False if self.t < len(self.trial_obs) else True
        self.info['current_t'] = self.t
        return np.array(self.trial_obs[self.t]).reshape(-1), reward, done, False, self.info


class IntervalTiming2D(gym.Env):

    metadata = {}

    def __init__(
            self,
            height,
            width,
            stimuli: list[int],
            scale: int = 1,
            timings: dict[str, Any] | None = None,
            rewards: dict[str, Any] | None = None,
            seed_offset: int = 0,
    ):

        self.height = height
        self.width = width
        self.stimuli = stimuli
        self.scale = scale
        self.timings = timings
        self.rewards = rewards
        self.seed_offset = seed_offset

        self.observation_space = gym.spaces.Box(0., 1., shape=(height, width), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

        self.timings['fixation'] = self.timings['fixation'] * self.scale
        self.timings['delay'] = self.timings['delay'] * self.scale
        self.timings['decision'] = self.timings['decision'] * self.scale
        self.fixation_obs = [np.zeros((self.height, self.width))] * self.timings['fixation']
        self.delay_obs = [np.zeros((self.height, self.width))] * self.timings['delay']
        self.decision_obs = [np.zeros((self.height, self.width))] * self.timings['decision']

        self.stim_obs = []
        self.sampled_interval = None
        self.trial_obs = None
        self.t = 0  # trial timestep
        self.info = None
        self.gt = None

    def in_period(self, name: str):
        if 0 < self.t <= self.timings['fixation']:
            period = 'fixation'
        elif self.timings['fixation'] < self.t <= (self.timings['fixation']+self.timings['stimulus']):
            period = 'stimulus'
        elif (self.timings['fixation']+self.timings['stimulus']) < self.t <= (self.timings['fixation']+self.timings['stimulus']+self.timings['delay']):
            period = 'delay'
        else:
            period = 'decision'

        if name == period:
            return True
        else:
            return False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed+self.seed_offset)
        #self.sampled_interval = self.stimuli[self.np_random.integers(0, len(self.stimuli), size=1, dtype=int)]
        self.sampled_interval = self.np_random.choice(self.stimuli)   # randomly choose a single value from self.stimuli
        self.timings['stimulus'] = self.sampled_interval*self.scale
        # form the trial observations
        self.stim_obs = [np.ones((self.height, self.width))]*self.scale+[np.zeros((self.height, self.width))]*(self.timings['stimulus']-2*self.scale)+[np.ones((self.height, self.width))]*self.scale
        self.trial_obs = self.fixation_obs + self.stim_obs + self.delay_obs + self.decision_obs

        self.t = 0
        self.gt = 1 if self.sampled_interval >= self.stimuli[len(self.stimuli)//2] else 0
        self.info = {
            'trial_interval': self.sampled_interval,
            'period': 'fixation',
            'gt': self.gt,
            'current_t': self.t
            }
        return np.array(self.trial_obs[self.t]), self.info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        self.t += 1

        if self.in_period('fixation'):
            self.info['period'] = 'fixation'
        elif self.in_period('stimulus'):
            self.info['period'] = 'stimulus'
        elif self.in_period('delay'):
            self.info['period'] = 'delay'
        elif self.in_period('decision'):
            if action == self.gt:
                reward = self.rewards['correct']
            else:
                reward = self.rewards['incorrect']
            done = True
            self.info['period'] = 'decision'
            self.info['current_t'] = self.t
            return np.array(self.trial_obs[self.t]).reshape(-1), reward, done, False, self.info

        reward = 0.
        done = False if self.t < len(self.trial_obs) else True
        self.info['current_t'] = self.t
        return np.array(self.trial_obs[self.t]), reward, done, False, self.info