import gym
import os
import json
from gym import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters
from collections import deque

import numpy as np

# Available actions
ACTION_NOTHING = 0
ACTION_ADD_SMALL_VM = 1
ACTION_REMOVE_SMALL_VM = 2
ACTION_ADD_MEDIUM_VM = 3
ACTION_REMOVE_MEDIUM_VM = 4
ACTION_ADD_LARGE_VM = 5
ACTION_REMOVE_LARGE_VM = 6

address = os.getenv('CLOUDSIM_GATEWAY_HOST', 'cloudsimplus-gateway')
port = os.getenv('CLOUDSIM_GATEWAY_PORT', '25333')
parameters = GatewayParameters(address=address,
                               port=int(port),
                               auto_convert=True)
gateway = JavaGateway(gateway_parameters=parameters)
simulation_environment = gateway.entry_point


def to_string(java_array):
    return gateway.jvm.java.util.Arrays.toString(java_array)


def to_nparray(raw_obs):
    obs = list(raw_obs)
    return np.array(obs)


# Based on https://github.com/openai/gym/blob/master/gym/core.py
class ThreeSizeAppEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, **kwargs):
        self.OBSERVATION_HISTORY_LENGTH: int = int(kwargs.get('observation_history_length', "1"))
        self.NUM_OF_ACTIONS = 7
        self.NUM_OF_OBSERVATION_METRICS = 7
        # actions are identified by integers 0-n

        self.action_space = spaces.Discrete(self.NUM_OF_ACTIONS)

        # observation metrics - all within 0-1 range
        # "vmAllocatedRatioHistory",
        # "avgCPUUtilizationHistory",
        # "p90CPUUtilizationHistory",
        # "avgMemoryUtilizationHistory",
        # "p90MemoryUtilizationHistory",
        # "waitingJobsRatioGlobalHistory",
        # "waitingJobsRatioRecentHistory"
        shape = (1, self.OBSERVATION_HISTORY_LENGTH, self.NUM_OF_OBSERVATION_METRICS) if self.OBSERVATION_HISTORY_LENGTH > 1 else (self.NUM_OF_OBSERVATION_METRICS, self.OBSERVATION_HISTORY_LENGTH)
        self.observation_space = spaces.Box(low=0,
                                            high=1.0,
                                            shape=shape,
                                            dtype=np.float32)
        params = {
            'INITIAL_VM_COUNT': kwargs.get('initial_vm_count'),
            'SOURCE_OF_JOBS': 'PARAMS',
            'JOBS': kwargs.get('jobs_as_json', '[]'),
            'SIMULATION_SPEEDUP': kwargs.get('simulation_speedup', '1.0'),
            'SPLIT_LARGE_JOBS': kwargs.get('split_large_jobs', 'false'),
        }
        self.observation_history = self._init_observation_history()
        if 'queue_wait_penalty' in kwargs:
            params['QUEUE_WAIT_PENALTY'] = kwargs['queue_wait_penalty']

        self.simulation_id = simulation_environment.createSimulation(params)

    def step(self, action):
        if type(action) == np.int64:
            action = action.item()
        result = simulation_environment.step(self.simulation_id, action)
        reward = result.getReward()
        done = result.isDone()
        raw_obs = result.getObs()

        obs = self._get_updated_observation_history(raw_obs=raw_obs)
        return (
            obs,
            reward,
            done,
            {}
        )

    def reset(self):
        result = simulation_environment.reset(self.simulation_id)
        raw_obs = result.getObs()
        self.observation_history = self._init_observation_history()
        obs = self._get_updated_observation_history(raw_obs=raw_obs)
        return obs

    def render(self, mode='human', close=False):
        # result is a string with arrays encoded as json
        result = simulation_environment.render(self.simulation_id)
        arr = json.loads(result)
        if mode == 'ansi' or mode == 'human':
            if mode == 'human':
                print([ser[-1] for ser in arr])

            return result
        elif mode == 'array':
            return arr
        elif mode != 'ansi' and mode != 'human':
            return super().render(mode)

    def close(self):
        # close the resources
        simulation_environment.close(self.simulation_id)

    def seed(self):
        simulation_environment.seed(self.simulation_id)

    def _get_updated_observation_history(self, raw_obs):
        if self.OBSERVATION_HISTORY_LENGTH == 1:
            obs = to_nparray(raw_obs)
        else:
            self.observation_history.pop()
            self.observation_history.appendleft(list(raw_obs))
            obs = np.array(list(self.observation_history))[np.newaxis, :]
        return obs

    def _init_observation_history(self):
        return deque([[0.0 for _ in range(self.NUM_OF_OBSERVATION_METRICS)] for _ in range(self.OBSERVATION_HISTORY_LENGTH)])