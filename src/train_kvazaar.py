import os
from glob import glob
from multiprocessing import cpu_count
from subprocess import Popen
import argparse
import datetime
import logging
import logging.handlers
import tempfile
from time import CLOCK_REALTIME
from configparser import ConfigParser

from ray import init as ray_init
from ray.exceptions import RayError
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import ray.tune.logger as ray_logger

from kvazaar_gym.envs.kvazaar_env import Kvazaar
from custom_callbacks import MyCallBacks
import common_tasks

nCores = cpu_count()

def parse_args():
    """
    Method that manages command line arguments.
    """
    parser = argparse.ArgumentParser(description="Trainer for Kvazaar video encoder using RLLIB.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-r", "--restore", action='store_true',help="Restore an agent from last checkpoint. Train from there.")
    ##required
    # parser.add_argument("-n", "--name", required=True, help="Name of the trainiing. This will be the name of the path for saving checkpoints.")
    # parser.add_argument("-i", "--iters", type=int, help="Number of training iterations", required=True)
    # parser.add_argument("-m", "--mode", help="Mode of video selection", choices=["random","rotating"], required=True)
    # parser.add_argument("-r", "--rewards", required=True, help="Path of rewards file")

    # #optional
    # parser.add_argument("-b", "--batch", type=int, help="Training batch size", default=200)
    # parser.add_argument("--mini_batch", type=int, help="Size of SGD minibatch", default=128)
    # parser.add_argument("-k", "--kvazaar", help="Kvazaar's executable file location", default= os.path.expanduser("~/malleable_kvazaar/bin/./kvazaar"))
    # parser.add_argument("-v", "--videos", help= "Path of the set of videos for training", default= os.path.expanduser("~/videos_kvazaar_train/"))
    # parser.add_argument("-c", "--cores", nargs=2, metavar=('start', 'end'), type=int, help= "Kvazaar's dedicated CPUS (range)", default=[0, int(nCores/2)-1])
    
    args = parser.parse_args()
    return args

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

def checkconf(conf):
    """Checker for configuration file options."""

    rewards = conf['common']['rewards']
    kvazaar = conf['common']['kvazaar']
    cores = conf['common']['cores'].split(",")
    cores[0] = int(cores[0])
    cores[1] = int(cores[1])
    batch = int(conf['train']['batch'])
    mini_batch = int(conf['train']['mini_batch'])
    videos = conf['train']['videos']
    mode = conf['train']['mode']
    iters = int(conf['train']['iters'])
    name = conf['train']['name']

    assert os.path.exists(rewards) , "La ruta de recompensas no existe"
    assert os.path.isfile(rewards) , "La ruta de recompensas no es un archivo"
    assert os.path.exists(kvazaar) , "La ruta de kvazaar no existe"
    assert cores[0] >= 0 and \
           cores[0] < nCores and \
           cores[1] >= 0 and \
           cores[1] < nCores and  \
           cores[0] < cores[1] , "La configuración de cores de kvazaar no es correcta"
    assert batch > 0 , "El tamaño de batch no es correcto"
    assert mini_batch > 0 , "El tamaño de batch no es correcto"
    assert mini_batch < batch , "El tamaño de mini_batch no es menor que el tamaño de batch"
    assert os.path.exists(videos), "La ruta de videos de entrenamiento no existe"
    assert mode == "random" or mode == "rotating", "El modo de entrenamiento no es random o rotating"
    assert iters > 0, "El número de iteraciones debe ser positivo"
    assert name != "" , "El entrenamiento debe tener un nombre"

    
def main():
    
    args = parse_args()

    # get configuration
    conf = ConfigParser()
    print(os.getcwd())
    conf.read('src/config.ini')
        
    fecha = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    #init path for results if not existing
    if not os.path.exists("./resultados/"):
        if(args.restore):
            raise FileExistsError("Training path not existing. You cannot restore.")
        else: os.makedirs("./resultados")
    
    training_name = conf['train']['name']
    #init path for results of this training if not existing
    results_path =  "resultados/" +  training_name + "/"
    if not os.path.exists(results_path):
        if(args.restore):
            raise FileExistsError("Training path not existing. You cannot restore.")
        else: os.makedirs(results_path)

    #create logger for video usage
    video_logger = get_video_logger(results_path + 'video_' + training_name + "_" + fecha + '.log')

    #create map_rewards
    rewards = common_tasks.create_map_rewards(conf['common']['rewards'])

    ##kvazaar options
    kvazaar_path = conf['common']['kvazaar']
    vids_path_train = conf['train']['videos']
    conf_cores = conf['common']['cores'].split(",")
    kvazaar_cores = [x for x in range(int(conf_cores[0]), int(conf_cores[1])+1)] 
    kvazaar_mode = conf['train']['mode']

    ##Set affinity of main process using cores left by kvazaar
    common_tasks.set_affinity(kvazaar_cores)

    # init directory in which to save checkpoints
    chkpt_root = results_path + "checkpoints/"
    

    # start Ray -- add `local_mode=True` here for debugging
    ray_init(ignore_reinit_error=True, local_mode=True)

    
    # register the custom environment
    select_env = "kvazaar-v0"
    
    n_iters = int(conf['train']['iters'])
    batch = int(conf['train']['batch'])
    mini_batch = int(conf['train']['mini_batch'])

    register_env(select_env, lambda config: Kvazaar(kvazaar_path=kvazaar_path, 
                                                        vids_path=vids_path_train, 
                                                        cores=kvazaar_cores,
                                                        mode=kvazaar_mode,
                                                        logger=video_logger,
                                                        num_steps=batch*n_iters,
                                                        batch=batch,
                                                        kvazaar_output=False,
                                                        rewards_map=rewards
                                                        ))


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"    
    config["num_workers"] = 1
    config["train_batch_size"] = batch
    config["rollout_fragment_length"] = batch
    config["sgd_minibatch_size"] = mini_batch
    config["callbacks"] = MyCallBacks


    #Create o restore logger for ray_results
    def get_ray_results_logger(config=config, name=training_name, results_path=results_path, fecha=fecha, restore=args.restore):
        logdir = None
        if restore:
            #get directory of ray_results 
            logdir = glob(results_path + name + '*')[0]
        else:
            logdir = tempfile.mkdtemp(prefix=name+"_"+fecha+"_", dir=results_path)
        
        return ray_logger.UnifiedLogger(config, logdir, loggers=None)

    ray_results_logger = get_ray_results_logger
    agent = ppo.PPOTrainer(config, env=select_env, logger_creator=ray_results_logger)

    status = "\033[1;32;40m{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}\033[0m"
    
    if args.restore:
        agent.restore(common_tasks.load_checkpoint(chkpt_root))
    
    # train a policy with RLlib using PPO
    for n in range(n_iters):
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
        except RayError:
            agent.stop()
            print("Training stopped.") 
            exit()
        except KeyboardInterrupt:
            agent.stop()
            print("Training stopped.") 
            exit()
    
    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

 
if __name__ == "__main__":
    try:
        main()
    except FileExistsError as e:
        print(e)
