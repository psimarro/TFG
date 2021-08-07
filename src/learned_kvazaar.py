import os
import multiprocessing
import glob
import getpass
import subprocess
import argparse
from configparser import ConfigParser

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
    parser = argparse.ArgumentParser(description="Ray agent launcher that restores a checkpoint training of Kvazaar gym model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    #Required
    parser.add_argument("-v", "--video", help= "Path of the tested video.", required=True)
    parser.add_argument("-p", "--path", required=True,type=str, help="Checkpoints path")

    #optional
    # parser.add_argument("-b", "--batch", type=int, help="Training batch size", default=200)
    # parser.add_argument("--mini_batch", type=int, help="Size of SGD minibatch", default=128)
    # parser.add_argument("-k", "--kvazaar", help="Kvazaar's executable file location.", default="/home/" + user + "/malleable_kvazaar/bin/./kvazaar")
    # parser.add_argument("-c", "--cores", nargs=2, metavar=('core_ini', 'core_fin'), type=int, help= "Kvazaar's dedicated CPUS (range)", default=[int(nCores/2), nCores-1])
    
    args = parser.parse_args()
    return args

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

def checkconf(conf):
    """Checker for configuration file options"""
    
    rewards = conf['common']['rewards']
    kvazaar = conf['common']['kvazaar']
    cores = conf['common']['cores'].split(",")
    cores[0] = int(cores[0])
    cores[1] = int(cores[1])

    assert os.path.exists(rewards) , "La ruta de recompensas no existe"
    assert os.path.isfile(rewards) , "La ruta de recompensas no es un archivo"
    assert os.path.exists(kvazaar) , "La ruta de kvazaar no existe"
    assert cores[0] >= 0 and \
           cores[0] < nCores and \
           cores[1] >= 0 and \
           cores[1] < nCores and  \
           cores[0] < cores[1] , "La configuración de cores de kvazaar no es correcta"

def main ():
    #get command line parameters
    args = parse_args()

    #get config
    conf = ConfigParser()
    conf.read('src/config.ini')
    checkconf(conf)

    ##kvazaar options
    kvazaar_path = conf['common']['kvazaar']
    vids_path_test = args.video
    conf_cores = list(conf['common']['cores'].split(","))
    kvazaar_cores = calcula_cores(int(conf_cores[0]),int(conf_cores[1]))
    
    ##Set affinity of main process using cores left by kvazaar
    set_affinity(kvazaar_cores)
    
    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True, local_mode=True)

    #configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"    
    config["num_workers"] = 0
    config["num_cpus_for_driver"] = nCores - len(kvazaar_cores)
    config["callbacks"] = MyCallBacks

    
    #create map_rewards
    rewards = create_map_rewards(conf['common']['rewards'])


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