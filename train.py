#!/usr/bin/env python
# encoding: utf-8

from gym_example.envs.kvazaar_env import Kvazaar_v0
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil
import multiprocessing


def main ():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    kvazaar_path = "/home/pedro/malleable_kvazaar/bin/./kvazaar"
    vid_path = "/home/pedro/Descargas/E_KristenAndSara_1280x720_60p.yuv"
    # register the custom environment
    select_env = "kvazaar-v0"
    register_env(select_env, lambda config: Kvazaar_v0(kvazaar_path=kvazaar_path, 
                   vid_path=vid_path, 
                   nCores=multiprocessing.cpu_count(),
                   intervalos=[25, 50, 100, 150]))


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    kvazaar_path = "/home/pedro/malleable_kvazaar/bin/./kvazaar"
    vid_path = "/home/pedro/Descargas/E_KristenAndSara_1280x720_60p.yuv"
    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env, kvazaar_path=kvazaar_path, 
                   vid_path=vid_path, 
                   nCores=multiprocessing.cpu_count(),
                   intervalos=[25, 50, 100, 150])
    
    state = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()
