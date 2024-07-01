#!/usr/bin/env python3

from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.sac.sac import SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from sb3_her_navigation.stage_env import StageEnv
from gymnasium.wrappers import TimeLimit
import rclpy

def main():
    rclpy.init()
    env = StageEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    trainer = SAC(policy='MultiInputPolicy', env=env, replay_buffer_class=HerReplayBuffer, learning_starts=10000)
    trainer.learn(total_timesteps=300000)
    trainer.policy.save(path='sb3_sac_her_stage')
    rclpy.spin(env.node)

if __name__ == '__main__':
    main()