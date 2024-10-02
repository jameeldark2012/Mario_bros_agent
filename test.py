from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


model = PPO(env = env ,policy = "MlpPolicy"  ,verbose=1)
model.learn(total_timesteps=100000 , log_interval=1)


mean_reward = evaluate_policy(model , env , 10 , render=True)
