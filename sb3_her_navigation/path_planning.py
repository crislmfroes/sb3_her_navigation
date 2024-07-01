#!/usr/bin/env python3

from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.sac.sac import SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from sb3_her_navigation.stage_env import StageEnv
from gymnasium.wrappers import TimeLimit
import rclpy
from threading import Thread


def main(args=None):
    rclpy.init(args=args)
    env = StageEnv()
    def run_node():
        rclpy.spin(env)
    thread2 = Thread(target=run_node)
    thread2.start()
    env.create_rate(frequency=1.0).sleep()
    env.execute_task(None, None)

if __name__ == '__main__':
    main()