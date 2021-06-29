from agents.base.agent import Agent

import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from skimage.color import rgb2gray


class QrmAgent(Agent):
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
        super(QrmAgent, self).__init__(
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
        )

    def select_action(self, state, goal_id, state_extra=None, actions=None):
        state = torch.as_tensor(state, device=self.device).float().unsqueeze(0)
        goal_id = torch.as_tensor([goal_id], device=self.device).float()
        if state_extra is not None:
            state_extra = torch.as_tensor([state_extra], device=self.device).float()
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device).float().unsqueeze(0)
        q_values = self.action_model(state, goal_id, state_extra, actions).detach()
        action = torch.argmax(q_values)
        return action.item()

    def update_action_model(self):
        if self.env_name == "two_keys":
            (
                state,
                action,
                reward,
                terminal,
                next_state,
                gnet_state,
                next_gnet_state,
            ) = self.action_replay_memory.retrieve()
            q = (
                self.action_model(state, gnet_state)
                .gather(-1, action.view(self.minibatch_size, 1))
                .squeeze(1)
            )
            a_max = (
                self.action_model(next_state, next_gnet_state).max(dim=1)[1].detach()
            )
            qmax = (
                self.clone_action_model(next_state, next_gnet_state)
                .max(dim=1)[0]
                .detach()
            )
        else:
            (
                reward,
                terminal,
                gnet_state,
                next_gnet_state,
                indexes,
            ) = self.action_replay_memory.retrieve_frame_stack()
            (
                state,
                action,
                state_extra,
                next_state,
                next_state_extra,
            ) = self.action_replay_memory.get_frame_stack(indexes)
            q = (
                self.action_model(state, gnet_state, state_extra, action[:, :-1])
                .squeeze(1)
                .gather(1, action[:, -1].view(self.minibatch_size, 1))
                .squeeze(1)
            )
            a_max = (
                self.action_model(
                    next_state, next_gnet_state, next_state_extra, action[:, 1:]
                )
                .squeeze(1)
                .max(dim=1)[1]
                .detach()
            )
            qmax = (
                self.clone_action_model(
                    next_state, next_gnet_state, next_state_extra, action[:, 1:]
                )
                .squeeze(1)
                .detach()
                .gather(1, a_max.view(self.minibatch_size, 1))
                .squeeze(1)
            )

        nonterminal_target = reward + self.gamma * qmax
        terminal_target = reward

        target = (
            terminal.float() * terminal_target
            + (~terminal).float() * nonterminal_target
        )

        loss = self.loss(q, target)

        self.action_opt.zero_grad()
        loss.backward()
        self.action_opt.step()

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
        floor_idx=-1,
        seed=-1,
    ):
        if not eval:
            self.frame = start_frame
            self.episode = episode
        result_data = dict(episode=[], reward=[], steps=[], frames=[])

        reward_history = []
        steps_history = []

        subgoal_progress = {}
        for key in self.gnet_manager.gnet_goals.keys():
            subgoal_progress[key] = []

        if load:
            self.load()

        progress_bar = tqdm(range(episode, episodes), unit="episode", disable=eval)

        if self.env_name == "four_rooms_3d":
            channels = 3
            img_height = 60
            img_width = 80
        elif self.env_name == "ai2thor_kitchen":
            channels = 4
            img_height = 100
            img_width = 100
        state_extra_size = 8

        for episode in progress_bar:
            steps = 0

            next_gnet_state = "start"
            if self.env_name == "ai2thor_kitchen":
                obs = self.env.reset(train=train, idx=floor_idx, seed=seed)
            else:
                obs = self.env.reset()
            state = self.process_obs(obs)
            if self.env_name == "two_keys":
                obs = state
            self.gnet_manager.reset()

            state_extra_stack = None
            state_stack = None
            action_stack = None

            if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
                state_extra_stack = np.zeros((state_extra_size * self.frame_stack))
                state_stack = np.zeros(
                    (channels * self.frame_stack, img_height, img_width)
                )
                action_stack = np.zeros((self.frame_stack - 1))

            episode_done = False
            total_reward = 0

            while not episode_done:
                self.gnet_manager.gnet_state = next_gnet_state

                if (
                    self.env_name == "four_rooms_3d"
                    or self.env_name == "ai2thor_kitchen"
                ):
                    if self.env_name == "four_rooms_3d":
                        state_extra = [
                            self.env.agent.pos[0],
                            self.env.agent.pos[2],
                            self.env.goal.pos[0],
                            self.env.goal.pos[2],
                            self.env.yellow_subgoal.pos[0],
                            self.env.yellow_subgoal.pos[2],
                            self.env.blue_subgoal.pos[0],
                            self.env.blue_subgoal.pos[2],
                        ]
                    elif self.env_name == "ai2thor_kitchen":
                        state_extra = (
                            [
                                self.env.last_meta["agent"]["position"]["x"],
                                self.env.last_meta["agent"]["position"]["z"],
                            ]
                            + self.env.fridge_pos
                            + self.env.light_pos
                        )
                    state_stack = np.concatenate((state_stack[channels:], state))
                    state_extra_stack = np.concatenate(
                        (state_extra_stack[state_extra_size:], state_extra)
                    )

                avg_exp_rate = 0
                for gnet_state in self.gnet_manager.gnet_goals[
                    self.gnet_manager.gnet_state
                ]["goal_selection_options"]:
                    avg_exp_rate += self.gnet_manager.get_exploration_rate(
                        self.gnet_manager.gnet_state, gnet_state
                    )
                avg_exp_rate /= len(
                    self.gnet_manager.gnet_goals[self.gnet_manager.gnet_state][
                        "goal_selection_options"
                    ]
                )
                if train and np.random.uniform() < avg_exp_rate:
                    action = np.random.choice(self.num_actions)
                elif (
                    eval
                    and self.env_name == "four_rooms_3d"
                    and np.random.uniform() < self.eps_eval
                ):
                    action = np.random.choice(self.num_actions)
                else:
                    if self.env_name == "two_keys":
                        action = self.select_action(
                            state,
                            self.gnet_manager.gnet_goals[self.gnet_manager.gnet_state][
                                "code"
                            ],
                        )
                    else:
                        action = self.select_action(
                            state_stack,
                            self.gnet_manager.gnet_goals[self.gnet_manager.gnet_state][
                                "code"
                            ],
                            state_extra_stack,
                            action_stack,
                        )

                if (
                    self.env_name == "four_rooms_3d"
                    or self.env_name == "ai2thor_kitchen"
                ):
                    action_stack = np.append(action_stack[1:], action)

                new_obs, reward, episode_done, info = self.env.step(action)
                new_state = self.process_obs(new_obs)

                if self.env_name == "four_rooms_3d":
                    next_state_extra = [
                        self.env.agent.pos[0],
                        self.env.agent.pos[2],
                        self.env.goal.pos[0],
                        self.env.goal.pos[2],
                        self.env.yellow_subgoal.pos[0],
                        self.env.yellow_subgoal.pos[2],
                        self.env.blue_subgoal.pos[0],
                        self.env.blue_subgoal.pos[2],
                    ]
                elif self.env_name == "ai2thor_kitchen":
                    next_state_extra = (
                        [
                            self.env.last_meta["agent"]["position"]["x"],
                            self.env.last_meta["agent"]["position"]["z"],
                        ]
                        + self.env.fridge_pos
                        + self.env.light_pos
                    )

                if self.env_name == "two_keys":
                    new_obs = new_state

                # Only count the final goal reward in the total
                total_reward += reward
                next_gnet_state = self.gnet_manager.gnet_state
                next_subgoal_id = self.idx2goalnet.index(self.gnet_manager.gnet_state)

                for path in self.gnet_manager.goal_paths:
                    for gnet_state in self.gnet_manager.gnet_goals[
                        path.current_gnet_goal
                    ]["goal_selection_options"]:
                        if self.gnet_manager.check_goal_satisfied(gnet_state):
                            next_subgoal_id = self.idx2goalnet.index(gnet_state)
                            next_gnet_state = gnet_state
                            if next_gnet_state != "end":
                                reward = 1
                            break
                if train:
                    if self.env_name == "two_keys":
                        self.action_replay_memory.save(
                            obs,
                            action,
                            episode_done,
                            reward,
                            new_obs,
                            self.gnet_manager.gnet_goals[self.gnet_manager.gnet_state][
                                "code"
                            ],
                            self.gnet_manager.gnet_goals[next_gnet_state]["code"],
                        )
                    else:
                        self.action_replay_memory.save(
                            obs,
                            action,
                            episode_done,
                            reward,
                            new_obs,
                            self.gnet_manager.gnet_goals[self.gnet_manager.gnet_state][
                                "code"
                            ],
                            self.gnet_manager.gnet_goals[next_gnet_state]["code"],
                            state_extra,
                            next_state_extra,
                        )

                state = new_state
                obs = new_obs

                if not eval:
                    self.frame += 1
                steps += 1

                if (
                    train
                    and self.frame > self.min_buffer
                    and self.frame > self.minibatch_size
                ):
                    if self.frame % self.update_interval == 0:
                        self.update_action_model()

                    if self.frame % self.clone_interval == 0:
                        self.clone(self.clone_action_model, self.action_model, self.tau)

                if self.frame % 100 == 0:
                    progress_bar.set_description("frame = {}".format(self.frame))

                if train:
                    if self.gnet_manager.gnet_state != next_gnet_state:
                        subgoal_progress[next_gnet_state].append(1)
                        self.gnet_manager.update_success_rate(next_gnet_state, True)

                if self.gnet_manager.gnet_state != next_gnet_state:
                    self.gnet_manager.set_state(
                        self.gnet_manager.gnet_state, next_gnet_state
                    )

            if (
                train
                and self.gnet_manager.gnet_state == self.idx2goalnet[next_subgoal_id]
                and self.gnet_manager.gnet_state != "end"
            ):
                for gnet_state in self.gnet_manager.gnet_goals[
                    self.gnet_manager.gnet_state
                ]["goal_selection_options"]:
                    subgoal_progress[gnet_state].append(0)
                    self.gnet_manager.update_success_rate(gnet_state, False)
            reward_history.append(total_reward)
            steps_history.append(steps)
            if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
                self.action_replay_memory.update_trajectory_id()

            if not eval and episode % self.eval_every == 0 and episode > 0:
                self.action_model.eval()

                eval_rewards = []
                eval_steps = []
                if self.env_name == "ai2thor_kitchen":
                    for i in range(1, 31):
                        results = self.run(
                            1, train=False, eval=True, floor_idx=i, seed=1
                        )
                        eval_rewards.append(results["reward"][0])
                        eval_steps.append(results["steps"][0])
                else:
                    for i in range(self.eval_episodes):
                        results = self.run(1, train=False, eval=True)
                        eval_rewards.append(results["reward"][0])
                        eval_steps.append(results["steps"][0])
                result_data["episode"].append(episode)
                result_data["reward"].append(eval_rewards)
                result_data["steps"].append(eval_steps)
                result_data["frames"].append(self.frame)

                self.action_model.train()
                if do_print:
                    self.print_stats(
                        reward_history, steps_history, result_data, subgoal_progress
                    )

            elif eval:
                result_data["episode"].append(episode)
                result_data["reward"].append(total_reward)
                result_data["steps"].append(steps)

            if (
                save_checkpoints
                and episode % self.save_every == 0
                and episode > 0
                and not eval
            ):
                self.save(result_data)

        return result_data
