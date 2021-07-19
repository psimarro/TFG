import os
import multiprocessing
import glob
import getpass
import subprocess
import argparse

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import gym

from kvazaar_gym.envs.kvazaar_env import Kvazaar
from custom_callbacks import MyCallBacks

nCores = multiprocessing.cpu_count()

def parse_args():
    """
    Method that manages command line arguments.
    """
    user = getpass.getuser()
    parser = argparse.ArgumentParser(description="Ray agent launcher that restores a checkpoint training of Kvazaar gym model.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #Required
    parser.add_argument("-v", "--video", help= "Path of the tested video.", required=True)
    parser.add_argument("-p", "--path", required=True,type=str, help="Checkpoints path", default="resultados/")
    parser.add_argument("-r", "--rewards", required=True, help="Path of rewards file")
    
    #optional
    parser.add_argument("-b", "--batch", type=int, help="Training batch size", default=200)
    parser.add_argument("--mini_batch", type=int, help="Size of SGD minibatch", default=128)
    parser.add_argument("-k", "--kvazaar", help="Kvazaar's executable file location.", default="/home/" + user + "/malleable_kvazaar/bin/./kvazaar")
    parser.add_argument("-c", "--cores", nargs=2, metavar=('core_ini', 'core_fin'), type=int, help= "Kvazaar's dedicated CPUS (range)", default=[int(nCores/2), nCores-1])
    
    args = parser.parse_args()
    return args, parser

def calcula_cores(core_ini, core_fin):
    """Returns a list integers that matches the CPUS for Kvazaar"""
    print(core_ini, core_fin)
    return [x for x in range(core_ini,core_fin+1)] 

def set_affinity(kvazaar_cores):
    """
    Method that sets the affinity of the main process according to the cpus set for Kvazaar.
    """
    pid = os.getpid()
    print("Current pid: ", pid)
    total_cores = [x for x in range (nCores)]
    cores_main_proc = [x for x in total_cores if x not in kvazaar_cores]
    p = subprocess.Popen(["taskset", "-cp", ",".join(map(str,cores_main_proc)), str(pid)])
    p.wait()

def create_map_rewards(rewards_path):
    rewards = {}
    rewards_file = open(rewards_path)
    rewards_file = rewards_file.read().splitlines()
    for line in rewards_file:
        key, value = line.split(",")
        rewards[int(key)]= int(value)
    return rewards

def main ():

    args, parser = parse_args()

    ##kvazaar options
    kvazaar_path = args.kvazaar
    vids_path_test = args.video
    kvazaar_cores = calcula_cores(args.cores[0],args.cores[1])
    

    ##Set affinity of main process using cores left by kvazaar
    set_affinity(kvazaar_cores)
    
    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True, local_mode=True)

    #configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"    
    config["num_workers"] = 0
    config["num_cpus_for_driver"] = nCores - len(kvazaar_cores)
    config["train_batch_size"] = args.batch
    config["rollout_fragment_length"] =  args.batch
    config["sgd_minibatch_size"] = args.mini_batch
    config["callbacks"] = MyCallBacks

    
    #create map_rewards
    rewards = create_map_rewards(args.rewards)


    # register the custom environment
    select_env = "kvazaar-v0"
    register_env(select_env, lambda config: Kvazaar(kvazaar_path=kvazaar_path, 
                                                       vids_path=vids_path_test, 
                                                       cores=kvazaar_cores,
                                                       mode=None,
                                                       logger=None,
                                                       batch=None,
                                                       kvazaar_output=True,
                                                       rewards_map=rewards))

    agent = ppo.PPOTrainer(config, env=select_env)


    #restore checkpoint
    chkpnt_root = str(args.path)
    if(chkpnt_root[len(chkpnt_root)-1] != '/'): chkpnt_root += "/"
    chkpt_file = max(glob.iglob(chkpnt_root + "*/*[!.tune_metadata]", recursive=True) , key=os.path.getctime) ##retrieve last checkpoint path
    print(chkpt_file)
    print(('----------------------\n' +
            ' ---------------------\n' +
            'checkpoint loaded --   {:} \n'+
            '----------------------\n' +
            ' ---------------------\n').format(chkpt_file))                                                
    
    agent.restore(chkpt_file)
    env = gym.make(select_env, kvazaar_path=kvazaar_path, 
                            vids_path=vids_path_test, 
                            cores=kvazaar_cores,
                            mode=None,
                            logger=None,
                            rewards_map=rewards)
    
    state = env.reset()
    sum_reward = 0
    done = None
    cpus_used = [0] * len(kvazaar_cores)
    steps = 0

    done = False
    while not done:

        action = agent.compute_action(state)
        
        
        state, reward, done, info = env.step(action)
        sum_reward += reward
        
        done = info["kvazaar"] == "END"

        if done:
            # report at the end of each episode
            print('cumulative reward {} in {:} steps, cpus used {:}'.format(sum_reward, steps, cpus_used))
        else: 
            print("action:", action+1, "core(s)")
            env.render()
            cpus_used[action] += 1
            steps += 1
    
    agent.stop()
    
if __name__ == "__main__":
    
    main()