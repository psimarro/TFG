from subprocess import Popen, PIPE, DEVNULL
import time
import os
from bisect import bisect

import gym
from gym.utils import seeding
from gym.spaces import Discrete
import numpy as np

class Kvazaar (gym.Env):

    metadata = {
        "render.modes": ["human"]
    }

    def set_video_selection_mode(self):
        abs_path = self.vids_path
        dirpath = os.path.dirname(abs_path) + "/"
        if dirpath == abs_path: 
        #if os.path.isdir(self.vids_path):
            self.vids_list = os.listdir(self.vids_path) #list all vids in vids_path
            if self.mode == "rotating":
                self.vids_list = sorted(self.vids_list) #sort videos 
            elif self.mode is None: self.mode = "random"

        else:    
        #elif os.path.isfile(os.path.dirname(self.vids_path) and self.mode is None:
            self.mode = "file" 
            self.vids_list = [os.path.basename(abs_path)]
            self.vid_selected['name'] = self.vids_list[0]
            self.vid_selected['dir_pos'] = 0
            self.vids_path = dirpath
    
    def random_video_selection(self):   
        randomInt = self.np_random.randint(0, len(self.vids_list))
        newVideo = self.vids_list[randomInt]
        self.vid_selected['name'] = newVideo
        self.vid_selected['pos'] = randomInt
 
    def rotating_video_selection(self):
        self.vid_selected['pos'] = (self.vid_selected['pos'] + 1) % len(self.vids_list)
        new_video = self.vids_list[self.vid_selected['pos']]
        self.vid_selected['name'] = new_video


    def log_metrics(self):
        message = ("\n ------------- \n"
                   "Total total_steps: {}" +
                   "\n ------------- \n" +
                    "VIDEO USAGE \n").format(self.total_steps)

        ## Calculate the usage's percentage of each video used, using the number of times each video was chosen.
        self.video_usage = {key:round((value/self.total_steps*100), 0) 
                            for key, value in self.video_usage.items()}    
        
        ## Write the usage in the message
        for video, usage in self.video_usage.items():
            message += video + " = " + str(usage) + "%\n"

        if self.logger: self.logger.info(message)

    def __init__(self, **kwargs):
        # Recogemos los argumentos:
        # kvazaar_path: ruta donde está instalado kvazaar
        # vids_path: ruta de los vídeos que utilizará el entorno
        # cores: lista de cores para kvazaar
        self.kvazaar_path = kwargs.get("kvazaar_path")
        self.vids_path = kwargs.get("vids_path")
        self.cores = kwargs.get("cores") #Lista de los cores que utiliza el entorno
        self.mode = kwargs.get("mode") #Modo de seleccion de videos 
        self.logger = kwargs.get("logger")
        self.max_steps = kwargs.get("num_steps")
        self.batch = kwargs.get("batch")
        self.kvazaar_output = kwargs.get("kvazaar_output")
        self.rewards_map=kwargs.get("rewards_map")


        self.vids_list = None #for random or rotating mode
        
        self.action_space = Discrete(len(self.cores))
        self.observation_space = Discrete(9)
        self.kvazaar = None ##object for kvazaar subprocess
        
        self.vid_selected = {'name': None, 'pos': 0}
        self.set_video_selection_mode()
        
        #metrics
        self.video_usage = {video: 0 for video in self.vids_list}
        self.total_steps = 0

        self.seed() #Generate seed for random numbers
        self.state = None
        #self.reset()
        

    def reset(self):
       
        self.info = {"fps": 0, "reward": 0, "kvazaar": "running"}

        #Generate random action sample for state resetting
        if self.kvazaar is None:
            self.reset_kvazaar()
        
        if self.state is None: 
            self.state = self.observation_space.sample()
            self.calculate_reward()
  

        self.done = False
        self.info["reward"] = self.reward
        self.episode_steps = 0
        return self.state

    def reset_kvazaar(self):
        
        #self.mode == file is already managed
        if self.mode == "random": 
            self.random_video_selection()
        if self.mode == "rotating": 
            self.rotating_video_selection()          


        #log new video
        if self.logger: 
            self.logger.info(self.vid_selected['name'] 
                             + " , pos " + str(self.vid_selected['pos']))
        

        command = [self.kvazaar_path, 
                   "--input", self.vids_path + self.vid_selected['name'], 
                   "--output", "/dev/null", 
                   "--preset=ultrafast", 
                   "--qp=22", "--owf=0", 
                   "--threads=" + str(len(self.cores))]
        
        #apply taskset using the range of cores setted as argument
        command = ["taskset","-c",",".join([str(x) for x in self.cores])] + command
        
        # kvazaar process generation
        self.kvazaar = Popen(command, 
                                        stdin=PIPE, 
                                        stdout=PIPE, 
                                        stderr=2 if self.kvazaar_output else DEVNULL,
                                        universal_newlines=True, bufsize=1, 
                                        env={'NUM_FRAMES_PER_BATCH': '24'})
                                        
        print("{} selected".format(self.vid_selected["name"]))
        
        #make sure kvazaar process is up
        while not self.kvazaar:
             time.sleep(1)


    def call_kvazaar(self, action):
        """ Interaction with Kvazaar process passing the new action, i.e. the new numbers of cpus used for the next video block."""
        s = "nThs:" + str(action)
        self.kvazaar.stdin.write(s + "\n")
        output= self.kvazaar.stdout.readline().strip()
        return output
    
    def step(self, action):
        if self.done:
            pass
        else:
            assert self.action_space.contains(action)
            action += 1 
            
            output = self.call_kvazaar(action)

            self.info["kvazaar"] = "running"
            #Check if kvazaar is done
            if(output == "END"):
                self.reset_kvazaar()
                output = self.call_kvazaar(action)
                self.info["kvazaar"] = "END"
                #self.done = True

            ##update metrics
            self.total_steps += 1
            self.episode_steps += 1
            self.video_usage[self.vid_selected['name']] += 1
            
            self.calculate_state(output=output)

            #log this num step and the video used
            if self.logger: 
                self.logger.info("Step {} : fps {}, action {}".format(str(self.total_steps),
                                                                str(self.info.get("fps")),
                                                                str(action)))
            
            if self.total_steps == self.max_steps: 
                #reached end of traininng
                self.log_metrics()

            if self.episode_steps == 50 :
                  self.done = True
        
        try:
            assert self.observation_space.contains(self.state) #check if new state is valid
        except AssertionError:
            print("INVALID STATE", self.state)

        return [self.state, self.calculate_reward(), self.done, self.info]
    
    def calculate_reward(self):
        self.reward = self.rewards_map.get(self.state)
        self.info["reward"] = self.reward

        return self.reward

    def calculate_state(self, output):
        ##Erase "FPS:" from output and save it in the new state.
        if (output == "END"):
            self.state = self.state
        else: 
            output_value = np.float32(output[4:])
            self.info["fps"] = '{:.2f}'.format(output_value)
            intervals = [10,16,20,24,27,30,35,40]
            states = [np.int(x) for x in range(9)]
            self.state = states[bisect(intervals, output_value)]

            # if output_value < 10: self.state = np.int64(0)
            # elif output_value < 16: self.state = np.int(1)
            # elif output_value < 20: self.state = np.int(2)
            # elif output_value < 24: self.state = np.int(3)
            # elif output_value < 27: self.state = np.int64(4)
            # elif output_value < 30: self.state = np.int64(5)
            # elif output_value < 35: self.state = np.int64(6)
            # elif output_value < 40: self.state = np.int64(7) 
            # else: self.state = np.int64(8)
    
    
    def render(self, mode="human"):
        l = 'obs:{}  fps:{:>1}  reward:{:<10}'.format(self.state, self.info["fps"], self.reward)
        print(l)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def close(self):
       if self.kvazaar:
           self.kvazaar.kill()
