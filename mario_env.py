from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.spaces.box import Box
import numpy as np
import cv2

def wrap_mario_env(env_):
    # env_ = gym_super_mario_bros.make('SuperMarioBros-v2')
    env_ = JoypadSpace(env_, SIMPLE_MOVEMENT)
    return env_

class MarioEnv(gym_super_mario_bros.SuperMarioBrosEnv):
    def __init__(self) -> None:
        super(MarioEnv, self).__init__()
 # Store the environment returned by make_env
        # Modify the observation space
        self.resolution = [96,96]
        self.observation_space = Box(low=0, high=255, shape=(self.resolution[0], self.resolution[1], 3), dtype=np.uint8)

    def reset(self):
        obs = super(MarioEnv, self).reset()
        obs = cv2.resize(obs , (96, 96))
        return obs

    def step(self, action):
        obs, reward, done , info  =super(MarioEnv, self).step(action)
        obs = cv2.resize(obs , (self.resolution[0], self.resolution[1]))
        return (obs,reward,done,info)

    # def render(self, mode="human"):
    #     return self.env.render(mode)

    # def close(self):
    #     return self.env.close()

