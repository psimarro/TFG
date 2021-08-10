from typing import Dict
import numpy as np

import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks



class MyCallBacks(DefaultCallbacks):
    def __init__(self, **kwargs):
        self.batch_fps = []
        self.batch_errors = {'above': 0, 'below': 0, 'total': 0,}
        self.episode_errors = {'above': 0, 'below': 0, 'total': 0}

        self.BELOW_ERROR_THRESHOLD = 20
        self.ABOVE_ERROR_THRESHOLD = 30

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        print("episode {} started".format(episode.episode_id))
        kvazaar_env = base_env.get_unwrapped()[0]
        video = getattr(kvazaar_env, "vid_selected")["name"]
        print("New video selected: " + video)
        episode.user_data["fps"] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        kvazaar_env = base_env.get_unwrapped()[0]
        fps = float(getattr(kvazaar_env, "info")["fps"])
        reward = int(getattr(kvazaar_env, "info")["reward"])
        steps = int(getattr(kvazaar_env, "total_steps"))
        batch = int(getattr(kvazaar_env, "batch"))

        last_action = episode.last_action_for()

        print("\tfps: {:.2f}, action: {}, reward: {}".format(fps, last_action, reward))
        episode.user_data["fps"].append(fps)
        self.batch_fps.append(fps)

        if fps < self.BELOW_ERROR_THRESHOLD: 
            self.batch_errors["below"] += 1
            self.episode_errors["below"] += 1
            self.episode_errors["total"] += 1
            self.batch_errors["total"] += 1

        if fps > self.ABOVE_ERROR_THRESHOLD: 
            self.batch_errors["above"] += 1
            self.episode_errors["above"] += 1
            self.episode_errors["total"] += 1
            self.batch_errors["total"] += 1

        if steps != 0 and (steps % batch) == 0 :
            mean_batch_fps = np.mean(self.batch_fps)
            episode.custom_metrics["batch_fps"] = mean_batch_fps
            episode.custom_metrics["batch_errors_above"] = self.batch_errors["above"]
            episode.custom_metrics["batch_errors_below"] = self.batch_errors["below"]
            episode.custom_metrics["total_batch_errors"] = self.batch_errors["total"]
            episode.custom_metrics["batch_error_ratio"] = (self.batch_errors["total"]/batch )*  100
            self.batch_fps = []
            self.batch_errors = {'above': 0, 'below': 0, 'total': 0}

        


    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        mean_fps = np.mean(episode.user_data["fps"])
        print("episode {} ended with length {}, mean fps {:.2f} and {} errors".format(
            episode.episode_id, episode.length, mean_fps, self.episode_errors))
        episode.custom_metrics["episode_fps"] = mean_fps
        episode.custom_metrics["episode_errors_above"] = self.episode_errors['above']
        episode.custom_metrics["episode_errors_below"] = self.episode_errors['below']
        episode.custom_metrics["total_episode_errors"] = self.episode_errors['total']
        episode.custom_metrics["episode_error_ratio"] = (self.episode_errors["total"]/episode.length) * 100
        self.episode_errors = {'above': 0, 'below': 0, 'total':0}      

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, trainer, result: dict, **kwargs):
        # print("trainer.train() result: {} -> {} episodes".format(
        #     trainer, result["episodes_this_iter"]))
        # # you can mutate the result dict to add new fields to return
        # result["callback_ok"] = True
        print("\nTotal time: {:.2f} s\nTotal episodes: {}\n".format(trainer._time_total,
                                                                  trainer._episodes_total))
            

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass