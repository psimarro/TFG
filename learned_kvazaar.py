from gym_example.envs.kvazaar_env import Kvazaar_v0
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import multiprocessing
import glob
import getpass
import subprocess
import argparse


nCores = multiprocessing.cpu_count()

def main ():

    args, parser = parse_args()

    ##kvazaar options
    kvazaar_path = args.kvazaar
    vids_path_test = args.videos
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
    config["train_batch_size"] = 200
    config["rollout_fragment_length"] =  200

    
    # register the custom environment
    select_env = "kvazaar-v0"
    register_env(select_env, lambda config: Kvazaar_v0(kvazaar_path=kvazaar_path, 
                                                       vids_path=vids_path_test, 
                                                       cores=kvazaar_cores,
                                                       mode=None))

    agent = ppo.PPOTrainer(config, env=select_env)


    #restore checkpoint
    chkpnt_root = args.path
    print(chkpnt_root)
    if(chkpnt_root[len(chkpnt_root)-1] != '/'): chkpnt_root += "/"
    if(chkpnt_root == parser.get_default("-p") or chkpnt_root == "tmp/exa/"): chkpnt_root += "*"
    print(parser.get_default("-p"))
    print(parser)
    print(chkpnt_root)
    chkpt_file = max(glob.iglob(chkpnt_root + "/*/*[!.tune_metadata]") , key=os.path.getctime) ##retrieve last checkpoint path
    
    print(('----------------------\n' +
            ' ---------------------\n' +
            'checkpoint loaded --   {:} \n'+
            '----------------------\n' +
            ' ---------------------\n').format(chkpt_file))                                                
    agent.restore(chkpt_file)
    env = gym.make(select_env, 
                kvazaar_path=kvazaar_path, 
                vids_path=vids_path_test, 
                cores=kvazaar_cores)
    
    state = env.reset()
    sum_reward = 0
    done = None
    cpus_used = [0] * len(kvazaar_cores)
    steps = 0

    while not done:
        action = agent.compute_action(state)
        print("action:", action+1, "core(s)")
        
        state, reward, done, info = env.step(action)
        sum_reward += reward
        
        env.render()

        if done:
            # report at the end of each episode
            print('cumulative reward {:.3f} in {:} steps, cpus used {:}'.format(sum_reward, steps, cpus_used))
        else: 
            cpus_used[action] += 1
            steps += 1
    
    env.close() # close kvazaar
    


def parse_args():
    user = getpass.getuser()
    parser = argparse.ArgumentParser(description="Ray agent launcher that restores a checkpoint training of Kvazaar gym model.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kvazaar", help="Kvazaar's executable file location.", default="/home/" + user + "/malleable_kvazaar/bin/./kvazaar")
    parser.add_argument("-v", "--videos", help= "Path of the tested video. If it is a directory, a random one will be choosed.", default= "/home/" + user + "/videos_kvazaar_test/")
    parser.add_argument("-c", "--cores", nargs=2, metavar=('core_ini', 'core_fin'), type=int, help= "Kvazaar's dedicated CPUS (range)", default=[int(nCores/2), nCores-1])
    parser.add_argument("-p", "--path", type=str, help="Checkpoints path", default="tmp/exa/")
    args = parser.parse_args()
    return args, parser

def calcula_cores(core_ini, core_fin):
    print(core_ini, core_fin)
    return [x for x in range(core_ini,core_fin+1)] 

def set_affinity(kvazaar_cores):
    pid = os.getpid()
    print("Curren pid: ", pid)
    total_cores = [x for x in range (nCores)]
    cores_main_proc = [x for x in total_cores if x not in kvazaar_cores]
    p = subprocess.Popen(["taskset", "-cp", ",".join(map(str,cores_main_proc)), str(pid)])
    p.wait()

if __name__ == "__main__":
    
    main()