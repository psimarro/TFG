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
import getpass
import sys
import subprocess


nCores = multiprocessing.cpu_count()

def main ():
    ##kvazaar options
    user = getpass.getuser()
    kvazaar_path = "/home/" + user + "/malleable_kvazaar/bin/./kvazaar"
    vids_path_train = "/home/" + user + "/videos_kvazaar/"
    vids_path_test = "/home/" + user + "/videos_train_test/"
    kvazaar_cores = [x for x in range(int(nCores/2),int(nCores))]

    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True,local_mode=True)

    
    # register the custom environment
    select_env = "kvazaar-v0"
    register_env(select_env, lambda config: Kvazaar_v0(kvazaar_path=kvazaar_path, 
                                                       vids_path=vids_path_train, 
                                                       cores=kvazaar_cores))


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"    
    config["num_workers"] = 0
    config["num_cpus_for_driver"] = int(nCores/2)
    config["train_batch_size"] = 200
    config["rollout_fragment_length"] =  200
         

    agent = ppo.PPOTrainer(config, env=select_env)

    status = "\033[1;32;40m{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}\033[0m"
    n_iter = int(sys.argv[1])

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        try:
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
        except KeyboardInterrupt:
            agent.stop()


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    #agent.cleanup() ##clean worker so kvazaar instances are shut down
    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env, 
                   kvazaar_path=kvazaar_path, 
                   vids_path=vids_path_test, 
                   cores=kvazaar_cores)
    
    state = env.reset()
    sum_reward = 0
    done = None
    cpus_used = [0] * len(kvazaar_cores)
    
    while not done:
        action = agent.compute_action(state)
        print("action:", action+1, "core(s)")
        cpus_used[action] += 1
        state, reward, done, info = env.step(action)
        sum_reward += reward

        if not done:
            env.render()
        else:
            # report at the end of each episode
            print("cumulative reward", sum_reward, "cpus used:", cpus_used)
    
    env.close() # close kvazaar
   


if __name__ == "__main__":
    id = os.getpid()
    print(id)
    cores_main = [x for x in range (int(nCores/2))]
    p = subprocess.Popen(["taskset", "-cp", ",".join(map(str,cores_main)), str(id)])
    p.wait()
    main()

