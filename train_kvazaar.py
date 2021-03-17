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
import argparse
import datetime


nCores = multiprocessing.cpu_count()

def main ():

    args = parse_args()

    ##kvazaar options
    kvazaar_path = args.kvazaar
    vids_path_train = args.videos
    kvazaar_cores = calcula_cores(args.cores[0],args.cores[1])
    kvazaar_mode = args.mode

    ##Set affinity of main process using cores left by kvazaar
    set_affinity(kvazaar_cores)

    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa/" + args.name + "/"
    # shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    # ray_results = "{}/ray_results/{}/".format(os.getenv("HOME"), args.name)
    # shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True,local_mode=True)

    
    # register the custom environment
    select_env = "kvazaar-v0"
    register_env(select_env, lambda config: Kvazaar_v0(kvazaar_path=kvazaar_path, 
                                                       vids_path=vids_path_train, 
                                                       cores=kvazaar_cores,
                                                       mode=kvazaar_mode))


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"    
    config["num_workers"] = 0
    config["num_cpus_for_driver"] = nCores - len(kvazaar_cores)
    config["train_batch_size"] = 200
    config["rollout_fragment_length"] =  200
         

    agent = ppo.PPOTrainer(config, env=select_env)

    status = "\033[1;32;40m{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}\033[0m"
    n_iter = args.iters

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
        except StopIteration:
            print("Training stopped.") 
            exit()

 

def parse_args():
    user = getpass.getuser()
    parser = argparse.ArgumentParser(description="Trainer for Kvazaar video encoder using RLLIB.",
    argument_default=argparse.SUPPRESS, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--iters", type=int, help="Number of training iterations", required=True)
    parser.add_argument("-b" "--batch", type=int, help="Size of training batch", default=200)
    parser.add_argument("-m", "--mode", help="Mode of video selection", choices=["random","rotating"], required=True)
    parser.add_argument("-k", "--kvazaar", help="Kvazaar's executable file location", default="/home/" + user + "/malleable_kvazaar/bin/./kvazaar")
    parser.add_argument("-v", "--videos", help= "Path of the set of videos for training", default= "/home/" + user + "/videos_kvazaar_train/")
    parser.add_argument("-c", "--cores", nargs=2, metavar=('start', 'end'), type=int, help= "Kvazaar's dedicated CPUS (range)", default=[int(nCores/2), nCores-1])
    parser.add_argument("-n", "--name", help="Name of the trainiing. This will be the name of the path for saving checkpoints.", default=datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    args = parser.parse_args()
    return args

def calcula_cores(core_ini, core_fin):
    print(core_ini, core_fin)
    return [x for x in range(core_ini,core_fin+1)] 

def set_affinity(kvazaar_cores):
    pid = os.getpid()
    print("Current pid: ", pid)
    total_cores = [x for x in range (nCores)]
    cores_main_proc = [x for x in total_cores if x not in kvazaar_cores]
    p = subprocess.Popen(["taskset", "-cp", ",".join(map(str,cores_main_proc)), str(pid)])
    p.wait()

if __name__ == "__main__":
    main()

