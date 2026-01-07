import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Minigrid'))
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import gymnasium as gym

# env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env = gym.make("MiniGrid-Custom", render_mode="human")
# env = gym.make("MiniGrid-Custom", render_mode="rgb_array")
# env = gym.make("MiniGrid-Fetch-8x8-N3-v0", render_mode="human")

env = ImgObsWrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(2e5)