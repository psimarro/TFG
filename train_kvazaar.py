import os
import multiprocessing
import getpass
import sys
import subprocess
import argparse
import datetime
import logging
import logging.handlers
import tempfile

import gym
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import ray.tune.logger as ray_logger

from kvazaar_gym.envs.kvazaar_env import Kvazaar
from custom_callbacks import MyCallBacks

nCores = multiprocessing.cpu_count()

def parse_args():
    """
    Method that manages command line arguments.
    """
    parser = argparse.ArgumentParser(description="Trainer for Kvazaar video encoder using RLLIB.",
    argument_default=argparse.SUPPRESS, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--iters", type=int, help="Number of training iterations", required=True)
    parser.add_argument("-b", "--batch", type=int, help="Training batch size", default=200)
    parser.add_argument("--mini_batch", type=int, help="Size of SGD minibatch", default=128)
    parser.add_argument("-m", "--mode", help="Mode of video selection", choices=["random","rotating"], required=True)
    parser.add_argument("-n", "--name", required=True, help="Name of the trainiing. This will be the name of the path for saving checkpoints.")
    parser.add_argument("-k", "--kvazaar", help="Kvazaar's executable file location", default= os.path.expanduser("~/malleable_kvazaar/bin/./kvazaar"))
    parser.add_argument("-v", "--videos", help= "Path of the set of videos for training", default= os.path.expanduser("~/videos_kvazaar_train/"))
    parser.add_argument("-c", "--cores", nargs=2, metavar=('start', 'end'), type=int, help= "Kvazaar's dedicated CPUS (range)", default=[0, int(nCores/2)-1])
    args = parser.parse_args()
    return args

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

def get_video_logger(video_log_file):
    """
    Method that returns a logger for the videos used.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.handlers.RotatingFileHandler(filename=video_log_file,mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger

def main():
    
    args = parse_args()
    fecha = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    #init path for results if not existing
    if not os.path.exists("./resultados/"):
        os.makedirs("./resultados")
    
    #init path for results of this training if not existing
    results_path =  "resultados/" + args.name + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    #create logger for video usage
    video_logger = get_video_logger(results_path + 'video_' + args.name + "_" + fecha + '.log')

    ##kvazaar options
    kvazaar_path = args.kvazaar
    vids_path_train = args.videos
    kvazaar_cores = [x for x in range(args.cores[0], args.cores[1]+1)] 
    kvazaar_mode = args.mode

    ##Set affinity of main process using cores left by kvazaar
    set_affinity(kvazaar_cores)

    # init directory in which to save checkpoints
    chkpt_root = results_path + "/checkpoints/"
    

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True,local_mode=True)

    
    # register the custom environment
    select_env = "kvazaar-v0"
    register_env(select_env, lambda config: Kvazaar(kvazaar_path=kvazaar_path, 
                                                        vids_path=vids_path_train, 
                                                        cores=kvazaar_cores,
                                                        mode=kvazaar_mode,
                                                        logger=video_logger,
                                                        num_steps=args.batch*args.iters,
                                                        batch=args.batch,
                                                        kvazaar_output=False
                                                        ))


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"    
    config["num_workers"] = 0
    config["num_cpus_for_driver"] = nCores - len(kvazaar_cores)
    config["train_batch_size"] = args.batch
    config["rollout_fragment_length"] = args.batch
    config["sgd_minibatch_size"] = args.mini_batch
    config["callbacks"] = MyCallBacks


    #Create logger for ray_results
    def get_ray_results_logger(config=config, name=args.name, results_path=results_path, fecha=fecha):
        logdir = tempfile.mkdtemp(
            prefix=name+"_"+fecha+"_", dir=results_path)
        return ray_logger.UnifiedLogger(config, logdir, loggers=None)

    ray_results_logger = get_ray_results_logger
    agent = ppo.PPOTrainer(config, env=select_env, logger_creator=ray_results_logger)

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
    
    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

 
if __name__ == "__main__":
    main()

