from gym_example.envs.kvazaar_env import Kvazaar_v0
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil
import multiprocessing
import glob

##kvazaar options
user = getpass.getuser()
kvazaar_path = "/home/" + user + "/malleable_kvazaar/bin/./kvazaar"
vids_path = "/home/" + user + "/videos_kvazaar/"
cpu_count = multiprocessing.cpu_count()

def main ():

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

# configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_workers"] = 1
    config["num_cpus_per_worker"] = cpu_count
    config["train_batch_size"] = 50
    config["rollout_fragment_length"] =  10
    config["sgd_minibatch_size"] = 4

    
    # register the custom environment
    select_env = "kvazaar-v0"
    register_env(select_env, lambda config: Kvazaar_v0(kvazaar_path=kvazaar_path, 
                                                       vids_path=vids_path, 
                                                       nCores=cpu_count))

    agent = ppo.PPOTrainer(config, env=select_env)

    chkpt_file = max(glob.iglob("./tmp/exa/*/*[!.tune_metadata]") , key=os.path.getctime)
    print(('----------------------\n' +
            ' ---------------------\n' +
            'checkpoint loaded --   {:} \n'+
            '----------------------\n' +
            ' ---------------------\n').format(chkpt_file))                                                
    agent.restore(chkpt_file)
    env = gym.make(select_env, 
                kvazaar_path=kvazaar_path, 
                vids_path=vids_path, 
                nCores=cpu_count)
    
    state = env.reset()
    sum_reward = 0
    done = None
    cpus_used = [0] * cpu_count
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
    


if __name__ == "__main__":
    main()