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
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        print("episode {} started".format(episode.episode_id))
        kvazaar_env = base_env.get_unwrapped()[0]
        video = getattr(kvazaar_env, "vid_selected")["name"]
        print("New video selected: " + video)
        episode.user_data["fps"] = []
        episode.hist_data["fps"] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        kvazaar_env = base_env.get_unwrapped()[0]
        fps = float(getattr(kvazaar_env, "info")["fps"])
        reward = int(getattr(kvazaar_env, "info")["reward"])
        kvazaar_info = getattr(kvazaar_env, "info")["kvazaar"]

        if kvazaar_info == "END":
            video = getattr(kvazaar_env, "vid_selected")["name"]
            print("New video selected: " + video)

        last_action = episode.last_action_for()

        print("\tfps: {:.2f}, action: {}, reward: {}".format(fps, last_action, reward))
        episode.user_data["fps"].append(fps)
        

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        mean_fps = np.mean(episode.user_data["fps"])
        print("episode {} ended with length {} and mean fps {:.2f}".format(
            episode.episode_id, episode.length, mean_fps))
        episode.custom_metrics["fps"] = mean_fps
        episode.hist_data["fps"] = episode.user_data["fps"]
        

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