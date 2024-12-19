# simple gymnasium env of interval discrimination for RL agents: which interval is longer (interval1 or interval2)
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType


class IntervalDisc1D(gym.Env):

    metadata = {}

    def __init__(
            self,
            height,
            width,
            stimuli,
            scale: int = 1,
            timings: dict[str, Any] | None = None,
            rewards: dict[str, Any] | None = None,
            seed_offset: int = 0,
    ):
        self.height = height
        self.width = width
        self.stimuli = stimuli  # list of tuples
        self.scale = scale
        self.timings = timings  # fixation, interval1, delay1, interval2, delay2, decision
        self.rewards = rewards
        self.seed_offset = seed_offset

        self.observation_space = gym.spaces.Box(0., 1., shape=(self.height,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

        #self.timings['fixation'] = self.timings['fixation'] * self.scale
        #self.timings['delay1'] = self.timings['delay1'] * self.scale
        #self.timings['delay2'] = self.timings['delay2'] * self.scale
        #self.timings['decision'] = self.timings['decision'] * self.scale
        self.timings['fixation'] = self.timings['fixation']
        self.timings['delay1'] = self.timings['delay1']
        self.timings['delay2'] = self.timings['delay2']
        self.timings['decision'] = self.timings['decision']
        self.fixation_obs = [0.] * self.timings['fixation']
        self.stim1_obs = []
        self.stim2_obs = []
        self.delay1_obs = [0.] * self.timings['delay1']
        self.delay2_obs = [0.] * self.timings['delay2']
        self.decision_obs = [0.] * self.timings['decision']
        self.sampled_interval1 = None
        self.sampled_interval2 = None
        self.trial_obs = None
        self.t = 0  # trial timestep
        self.info = None
        self.gt = None

    def in_period(self, name: str):
        if 0 < self.t <= self.timings['fixation']:
            period = 'fixation'
        elif self.timings['fixation'] < self.t <= (self.timings['fixation']+self.timings['stimulus1']):
            period = 'stimulus1'
        elif (self.timings['fixation']+self.timings['stimulus1']) < self.t <= (self.timings['fixation']+self.timings['stimulus1']+self.timings['delay1']):
            period = 'delay1'
        elif (self.timings['fixation']+self.timings['stimulus1']+self.timings['delay1']) < self.t <= (self.timings['fixation']+self.timings['stimulus1']+self.timings['delay1']+self.timings['stimulus2']):
            period = 'stimulus2'
        elif (self.timings['fixation']+self.timings['stimulus1']+self.timings['delay1']+self.timings['stimulus2']) < self.t <= (self.timings['fixation']+self.timings['stimulus1']+self.timings['delay1']+self.timings['stimulus2']+self.timings['delay2']):
            period = 'delay2'
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

        self.sampled_interval1, self.sampled_interval2 = self.np_random.choice(self.stimuli)
        self.timings['stimulus1'] = self.sampled_interval1*self.scale
        self.timings['stimulus2'] = self.sampled_interval2*self.scale

        # form the trial observations
        #self.stim_obs = [1.]*self.scale+[0.]*(self.timings['stimulus']-2*self.scale)+[1.]*self.scale
        self.stim1_obs = [1.] + [0.] * (self.timings['stimulus1'] - 2) + [1.]
        self.stim2_obs = [1.] + [0.] * (self.timings['stimulus2'] - 2) + [1.]
        self.trial_obs = self.fixation_obs + self.stim1_obs + self.delay1_obs + self.stim2_obs + self.delay2_obs + self.decision_obs

        self.t = 0
        self.gt = 1 if self.sampled_interval2 > self.sampled_interval1 else 0
        self.info = {
            'trial_intervals': (self.sampled_interval1, self.sampled_interval2),
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
        elif self.in_period('stimulus1'):
            self.info['period'] = 'stimulus1'
        elif self.in_period('delay1'):
            self.info['period'] = 'delay1'
        elif self.in_period('stimulus2'):
            self.info['period'] = 'stimulus2'
        elif self.in_period('delay2'):
            self.info['period'] = 'delay2'
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