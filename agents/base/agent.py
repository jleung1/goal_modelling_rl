import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
import copy
from tqdm import tqdm
from gym_minigrid.wrappers import *
import csv
import pickle


class Agent:
    def __init__(
        self,
        env,
        gnet_manager,
        action_model,
        action_memory,
        action_space,
        goal_space,
        num_goals,
        idx2goalnet,
        env_name,
        device,
    ):
        self.action_model = action_model
        self.action_replay_memory = action_memory

        self.env = env
        self.device = device
        self.gnet_manager = gnet_manager
        self.idx2goalnet = idx2goalnet
        self.env_name = env_name

        self.num_actions = action_space
        self.goal_space = goal_space
        self.num_goals = num_goals

        self.minibatch_size = 32
        self.gamma = 0.99
        self.save_every = 500

        self.eval_every = 100
        self.eval_episodes = 100
        self.action_lr = 0.0001

        self.frame_stack = 4

        if self.env_name == "two_keys":
            self.update_interval = 1
            self.clone_interval = 1
            self.min_buffer = 25000
            self.tau = 0.0001

        elif self.env_name == "four_rooms_3d":
            self.update_interval = 1
            self.clone_interval = 1000
            self.tau = 1
            self.min_buffer = 5000
            self.eps_eval = 0.05

        elif self.env_name == "ai2thor_kitchen":
            self.update_interval = 1
            self.clone_interval = 1000
            self.tau = 1
            self.min_buffer = 2000
            self.save_every = 50
            self.eval_every = 50
            self.eps_eval = 0.05

        self.save_dir = "./save/models_and_data"

        self.loss = nn.SmoothL1Loss()
        self.action_opt = optim.Adam(self.action_model.parameters(), lr=self.action_lr)

        self.clone_action_model = copy.deepcopy(action_model).eval().to(device)

        self.clone(self.clone_action_model, self.action_model, 1)

    def save(self, metadata):
        torch.save(self.action_model.state_dict(), self.save_dir + "/low_level.pt")
        torch.save(
            self.clone_action_model.state_dict(), self.save_dir + "/low_level_clone.pt"
        )
        torch.save(self.action_opt.state_dict(), self.save_dir + "/low_level_opt.pt")

        with open(
            self.save_dir + "/gnet_reward_results.csv", "w", newline=""
        ) as savefile:
            wr = csv.writer(savefile, quoting=csv.QUOTE_ALL)
            for i, ep in enumerate(metadata["episode"]):
                wr.writerow([ep] + metadata["reward"][i])

        with open(
            self.save_dir + "/gnet_step_results.csv", "w", newline=""
        ) as savefile:
            wr = csv.writer(savefile, quoting=csv.QUOTE_ALL)
            for i, ep in enumerate(metadata["episode"]):
                wr.writerow([ep] + metadata["steps"][i])

        with open(
            self.save_dir + "/gnet_frame_results.csv", "w", newline=""
        ) as savefile:
            wr = csv.writer(savefile, quoting=csv.QUOTE_ALL)
            for i, ep in enumerate(metadata["episode"]):
                wr.writerow([ep] + [metadata["frames"][i]])

        outfile = open(self.save_dir + "/goal_successes.pkl", "wb")
        pickle.dump(self.gnet_manager.goal_successes, outfile)
        outfile.close()

        self.action_replay_memory.save_to_disk()

    def process_obs(self, obs):
        if self.env_name == "two_keys":
            state = obs["image"].transpose(2, 1, 0)
        elif self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            state = obs.transpose(2, 0, 1)
            state = state / 255.0

        return state

    def print_stats(self, reward_history, steps_history, metadata, subgoal_progress):
        avg_return = np.array(metadata["reward"][-1]).mean()
        avg_steps = np.array(metadata["steps"][-1]).mean()

        avg_return_history = np.array(reward_history[-self.eval_every :]).mean()
        avg_steps_history = np.array(steps_history[-self.eval_every :]).mean()

        print(
            "\nAverage eval return ({} episodes): {:.2f}".format(
                self.eval_every, avg_return
            )
        )
        print(
            "Average eval steps ({} episodes): {:.2f}".format(
                self.eval_every, avg_steps
            )
        )
        print(
            "Average return ({} episodes): {:.2f}".format(
                self.eval_every, avg_return_history
            )
        )
        print(
            "Average steps ({} episodes): {:.2f}".format(
                self.eval_every, avg_steps_history
            )
        )
        print("Average subgoal return (last {} episodes):".format(self.eval_every))
        for key in subgoal_progress.keys():
            if len(subgoal_progress[key]) > 0:
                print(
                    "\t{}: \t{:.2f}".format(
                        key, np.array(subgoal_progress[key][-50:]).mean()
                    )
                )
        print("Exploration Rates: ")
        for key in subgoal_progress.keys():
            if "goal_selection_options" in self.gnet_manager.gnet_goals[key].keys():
                for goal in self.gnet_manager.gnet_goals[key]["goal_selection_options"]:
                    print(
                        "\tstart: {}\n\tgoal: {}: \t{}".format(
                            key, goal, self.gnet_manager.get_exploration_rate(key, goal)
                        )
                    )

    def clone(self, cloned_model, model, tau):
        for target_param, local_param in zip(
            cloned_model.parameters(), model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def run(
        self,
        episodes,
        train=False,
        load=False,
        eval=False,
        episode=0,
        start_frame=0,
        save_checkpoints=False,
        do_print=True,
    ):
        pass
