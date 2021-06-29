from agents.base.agent import Agent

import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from skimage.color import rgb2gray


class DaqnAgent(Agent):
    def __init__(
        self,
        env,
        gnet_manager,
        action_model,
        goal_q_table,
        action_memory,
        action_space,
        goal_space,
        num_goals,
        idx2goalnet,
        env_name,
        device,
    ):
        super(DaqnAgent, self).__init__(
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
        self.eps_hi = 0.5
        self.goal_q_table = goal_q_table
        self.goal_gamma = 0.9
        self.goal_lr = 0.01

    def save(self, result_data):
        super(DaqnAgent, self).save(result_data)

        outfile = open(self.save_dir + "/goal_q_table.npy", "wb")
        np.save(outfile, self.goal_q_table)
        outfile.close()

    def select_action(self, state, goal, goal_id, state_extra=None, actions=None):
        state = torch.as_tensor(state, device=self.device).float().unsqueeze(0)
        goal = torch.as_tensor([goal], device=self.device).float()
        goal_id = torch.as_tensor([goal_id], device=self.device).float()
        if state_extra is not None:
            state_extra = torch.as_tensor([state_extra], device=self.device).float()
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device).float().unsqueeze(0)
        q_values = self.action_model(
            state, goal, goal_id, state_extra, actions
        ).detach()
        action = torch.argmax(q_values)

        return action.item()

    def select_subgoal(self, state, mask):
        state_int = int("".join([str(x) for x in state]), 2)
        subgoal = np.array(self.goal_q_table[state_int] + mask).argmax()

        subgoal_coords = self.gnet_manager.get_goal_state(self.idx2goalnet[subgoal])[:2]

        return subgoal, subgoal_coords

    def update_action_model(self):
        if self.env_name == "two_keys":
            (
                state,
                action,
                reward,
                terminal,
                next_state,
                goal,
                target_gnet_state,
            ) = self.action_replay_memory.retrieve()
            q = (
                self.action_model(state, goal, target_gnet_state)
                .gather(1, action.view(self.minibatch_size, 1))
                .squeeze(1)
            )
            a_max = (
                self.action_model(next_state, goal, target_gnet_state)
                .max(dim=1)[1]
                .detach()
            )
            qmax = (
                self.clone_action_model(next_state, goal, target_gnet_state)
                .max(dim=1)[0]
                .detach()
            )
        else:
            (
                reward,
                terminal,
                goal,
                target_gnet_state,
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
                self.action_model(
                    state, goal, target_gnet_state, state_extra, action[:, :-1]
                )
                .squeeze(1)
                .gather(1, action[:, -1].view(self.minibatch_size, 1))
                .squeeze(1)
            )
            a_max = (
                self.action_model(
                    next_state, goal, target_gnet_state, next_state_extra, action[:, 1:]
                )
                .squeeze(1)
                .max(dim=1)[1]
                .detach()
            )
            qmax = (
                self.clone_action_model(
                    next_state, goal, target_gnet_state, next_state_extra, action[:, 1:]
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

    # For the tabular version, update using the experience from the episode
    def update_goal_model(self, experiences):
        for state, action, reward, terminal, next_state in experiences:
            state_int = int("".join([str(x) for x in state]), 2)
            next_state_int = int("".join([str(x) for x in next_state]), 2)
            if terminal:
                self.goal_q_table[state_int, action] = reward
            else:
                self.goal_q_table[state_int, action] = self.goal_q_table[
                    state_int, action
                ] + self.goal_lr * (
                    reward
                    + self.goal_gamma
                    * np.array(
                        self.goal_q_table[state_int]
                        + self.gnet_manager.gnet_goals[self.gnet_manager.gnet_state][
                            "mask"
                        ]
                    ).max()
                    - self.goal_q_table[state_int, action]
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
        floor_idx=-1,
        seed=-1,
    ):
        if not eval:
            self.frame = start_frame
            episode = episode
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
            if self.env_name == "ai2thor_kitchen":
                obs = self.env.reset(train=train, idx=floor_idx, seed=seed)
            else:
                obs = self.env.reset()
            state = self.process_obs(obs)
            if self.env_name == "two_keys":
                obs = state
            self.gnet_manager.reset()

            goal_data = []
            trajectory_data = []
            her_trajectory_data = []

            episode_done = False
            total_reward = 0

            state_extra_stack = None
            state_stack = None
            action_stack = None

            if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
                state_extra_stack = np.zeros((state_extra_size * self.frame_stack))
                state_stack = np.zeros(
                    (channels * self.frame_stack, img_height, img_width)
                )
                action_stack = np.zeros((self.frame_stack - 1))

            # Outer loop for high level
            while not episode_done:
                state_goal_selection = [
                    1 if x == True else 0
                    for x in self.gnet_manager.current_goal_state()
                ]
                state_goal_selection = state_goal_selection[2:] + [0]

                extrinsic_reward = 0

                if (
                    train
                    and self.env_name != "ai2thor_kitchen"
                    and np.random.uniform() < self.eps_hi
                ):
                    choices = self.gnet_manager.generate_options_list()
                    probs = []
                    target_choices = []
                    for init_g, target_g in choices:
                        success_rate = (
                            self.gnet_manager.goal_successes[init_g][target_g].sum()
                            / self.gnet_manager.last_num_goals
                            if len(self.gnet_manager.goal_successes[init_g][target_g])
                            > 0
                            else 0
                        )
                        probs.append(1.0 - success_rate + 0.1)
                        target_choices.append(target_g)

                    probs = probs / np.array(probs).sum()
                    chosen = np.random.choice(target_choices, p=probs)
                    subgoal_id = self.idx2goalnet.index(chosen)
                    subgoal_coord = self.gnet_manager.get_goal_state(chosen)[:2]

                    self.gnet_manager.gnet_state = self.gnet_manager.get_parent_goal(
                        chosen
                    )
                elif self.env_name == "ai2thor_kitchen":
                    chosen = self.gnet_manager.generate_options_list()[0][-1]
                    self.gnet_manager.gnet_state = self.gnet_manager.get_parent_goal(
                        chosen
                    )
                    subgoal_id = self.idx2goalnet.index(chosen)
                    subgoal_coord = self.gnet_manager.get_goal_state(chosen)[:2]
                else:
                    goalnet_code, mask = self.gnet_manager.generate_goal_mask()
                    subgoal_id, subgoal_coord = self.select_subgoal(
                        state_goal_selection, mask
                    )
                    self.gnet_manager.gnet_state = self.gnet_manager.get_parent_goal(
                        self.idx2goalnet[subgoal_id]
                    )

                subgoal_data = []
                subgoal_achieved = False
                other_subgoal_achieved = False
                achieved_subgoal_id = 0

                while (
                    not subgoal_achieved
                    and not episode_done
                    and not other_subgoal_achieved
                ):
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

                    if (
                        train
                        and np.random.uniform()
                        < self.gnet_manager.get_exploration_rate(
                            self.gnet_manager.gnet_state, self.idx2goalnet[subgoal_id]
                        )
                    ):
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
                                subgoal_coord,
                                self.gnet_manager.gnet_goals[
                                    self.idx2goalnet[subgoal_id]
                                ]["code"],
                            )
                        else:
                            action = self.select_action(
                                state_stack,
                                subgoal_coord,
                                self.gnet_manager.gnet_goals[
                                    self.idx2goalnet[subgoal_id]
                                ]["code"],
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

                    if self.env_name == "two_keys":
                        new_obs = new_state

                    next_state_extra = None
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
                    if self.gnet_manager.check_goal_satisfied(
                        self.idx2goalnet[subgoal_id]
                    ):
                        subgoal_achieved = True
                        subgoal_reward = 1
                    else:
                        # Check if any other goal state has been reached
                        for path in self.gnet_manager.goal_paths:
                            for gnet_state in self.gnet_manager.gnet_goals[
                                path.current_gnet_goal
                            ]["goal_selection_options"]:
                                if (
                                    self.gnet_manager.check_goal_satisfied(gnet_state)
                                    and gnet_state != "end"
                                ):
                                    other_subgoal_achieved = True
                                    achieved_parent_goal = path.current_gnet_goal
                                    achieved_subgoal_id = self.idx2goalnet.index(
                                        gnet_state
                                    )
                                subgoal_reward = 0
                    if train:
                        if episode_done or subgoal_achieved or other_subgoal_achieved:
                            done = True
                        else:
                            done = False

                        if self.env_name == "two_keys":
                            subgoal_data.append(
                                (
                                    obs,
                                    action,
                                    done,
                                    subgoal_reward,
                                    new_obs,
                                    self.gnet_manager.target_goal_state[:2],
                                    self.gnet_manager.gnet_goals[
                                        self.idx2goalnet[subgoal_id]
                                    ]["code"],
                                )
                            )
                            self.action_replay_memory.save(
                                obs,
                                action,
                                done,
                                subgoal_reward,
                                new_obs,
                                self.gnet_manager.target_goal_state[:2],
                                self.gnet_manager.gnet_goals[
                                    self.idx2goalnet[subgoal_id]
                                ]["code"],
                            )
                        else:
                            subgoal_data.append(
                                (
                                    obs,
                                    action,
                                    done,
                                    subgoal_reward,
                                    new_obs,
                                    self.gnet_manager.target_goal_state[:2],
                                    self.gnet_manager.gnet_goals[
                                        self.idx2goalnet[subgoal_id]
                                    ]["code"],
                                    state_extra,
                                    next_state_extra,
                                )
                            )
                            trajectory_data.append(
                                (
                                    obs,
                                    action,
                                    done,
                                    subgoal_reward,
                                    new_obs,
                                    self.gnet_manager.target_goal_state[:2],
                                    self.gnet_manager.gnet_goals[
                                        self.idx2goalnet[subgoal_id]
                                    ]["code"],
                                    state_extra,
                                    next_state_extra,
                                )
                            )

                    state = new_state
                    obs = new_obs
                    total_reward += reward
                    extrinsic_reward += reward
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
                            self.clone(
                                self.clone_action_model, self.action_model, self.tau
                            )

                    if self.frame % 100 == 0:
                        progress_bar.set_description("frame = {}".format(self.frame))

                if subgoal_achieved:
                    r_gnet_goal = self.gnet_manager.gnet_state
                    r_subgoal_id = subgoal_id
                elif (
                    self.env_name == "two_keys"
                    and self.env.agent_pos.tolist() == self.env.goal_pos.tolist()
                ):
                    r_gnet_goal = self.gnet_manager.gnet_state
                    r_subgoal_id = self.idx2goalnet.index("end")
                elif self.env_name == "four_rooms_3d" and self.env.near(self.env.goal):
                    r_gnet_goal = self.gnet_manager.gnet_state
                    r_subgoal_id = self.idx2goalnet.index("end")
                elif other_subgoal_achieved:
                    r_gnet_goal = achieved_parent_goal
                    r_subgoal_id = achieved_subgoal_id
                if train:
                    new_goal = self.gnet_manager.current_goal_state()[:2]
                    first = True
                    if self.env_name == "two_keys":
                        for (
                            r_state,
                            r_action,
                            r_done,
                            _,
                            r_new_state,
                            _,
                            r_target_gnet_state,
                        ) in subgoal_data[::-1]:
                            if first:
                                new_reward = 1
                                first = False
                            else:
                                new_reward = 0
                            self.action_replay_memory.save(
                                r_state,
                                r_action,
                                r_done,
                                new_reward,
                                r_new_state,
                                new_goal,
                                r_target_gnet_state,
                            )
                    else:
                        (
                            old_state,
                            old_action,
                            old_done,
                            old_reward,
                            old_new_state,
                            old_goal,
                            old_target_gnet_state,
                            old_state_extra,
                            old_next_state_extra,
                        ) = subgoal_data[-1]
                        subgoal_data[-1] = (
                            old_state,
                            old_action,
                            old_done,
                            1,
                            old_new_state,
                            old_goal,
                            old_target_gnet_state,
                            old_state_extra,
                            old_next_state_extra,
                        )
                        for (
                            r_state,
                            r_action,
                            r_done,
                            r_reward,
                            r_new_state,
                            _,
                            r_target_gnet_state,
                            state_extra,
                            next_state_extra,
                        ) in subgoal_data:
                            her_trajectory_data.append(
                                (
                                    r_state,
                                    r_action,
                                    r_done,
                                    r_reward,
                                    r_new_state,
                                    new_goal,
                                    r_target_gnet_state,
                                    state_extra,
                                    next_state_extra,
                                )
                            )
                    self.gnet_manager.update_success_rate(
                        self.idx2goalnet[subgoal_id], subgoal_achieved
                    )

                    if subgoal_achieved:
                        subgoal_progress[self.idx2goalnet[subgoal_id]].append(1)
                        if self.env_name != "ai2thor_kitchen":
                            new_state_goal_selection = [
                                1 if x == True else 0
                                for x in self.gnet_manager.current_goal_state()
                            ]
                            new_state_goal_selection = new_state_goal_selection[:2]

                            goal_data.append(
                                (
                                    state_goal_selection,
                                    subgoal_id,
                                    episode_done,
                                    extrinsic_reward,
                                    new_state_goal_selection,
                                )
                            )
                        self.gnet_manager.set_state(
                            r_gnet_goal, self.idx2goalnet[r_subgoal_id]
                        )
                    # Add a relabelled transition for the goal selection model
                    elif (
                        self.env_name == "two_keys"
                        and self.env.agent_pos.tolist() == self.env.goal_pos.tolist()
                    ) or (
                        self.env_name == "four_rooms_3d"
                        and self.env.near(self.env.goal)
                    ):
                        new_state_goal_selection = [
                            1 if x == True else 0
                            for x in self.gnet_manager.current_goal_state()
                        ]
                        new_state_goal_selection = new_state_goal_selection[:2] + [1]
                        goal_data.append(
                            (
                                state_goal_selection,
                                self.idx2goalnet.index("end"),
                                episode_done,
                                extrinsic_reward,
                                new_state_goal_selection,
                            )
                        )
                    elif other_subgoal_achieved:
                        subgoal_progress[self.idx2goalnet[subgoal_id]].append(0)
                        self.gnet_manager.set_state(
                            r_gnet_goal, self.idx2goalnet[r_subgoal_id]
                        )
                        if self.env_name != "ai2thor_kitchen":
                            new_state_goal_selection = [
                                1 if x == True else 0
                                for x in self.gnet_manager.current_goal_state()
                            ]
                            new_state_goal_selection = new_state_goal_selection[:2] + [
                                0
                            ]
                            goal_data.append(
                                (
                                    state_goal_selection,
                                    achieved_subgoal_id,
                                    episode_done,
                                    extrinsic_reward,
                                    new_state_goal_selection,
                                )
                            )
                    else:
                        subgoal_progress[self.idx2goalnet[subgoal_id]].append(0)

                elif (subgoal_achieved or other_subgoal_achieved) and not episode_done:
                    self.gnet_manager.set_state(
                        r_gnet_goal, self.idx2goalnet[r_subgoal_id]
                    )
                state_goal_selection = new_state

            if self.env_name != "ai2thor_kitchen":
                self.update_goal_model(goal_data)

            if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
                for (
                    r_state,
                    r_action,
                    r_done,
                    r_reward,
                    r_new_state,
                    r_target_goal,
                    r_target_gnet_state,
                    r_c_goal,
                    r_n_goal,
                ) in trajectory_data:
                    self.action_replay_memory.save(
                        r_state,
                        r_action,
                        r_done,
                        r_reward,
                        r_new_state,
                        r_target_goal,
                        r_target_gnet_state,
                        r_c_goal,
                        r_n_goal,
                    )
                self.action_replay_memory.update_trajectory_id()
                for (
                    r_state,
                    r_action,
                    r_done,
                    r_reward,
                    r_new_state,
                    r_target_goal,
                    r_target_gnet_state,
                    r_c_goal,
                    r_n_goal,
                ) in her_trajectory_data:
                    self.action_replay_memory.save(
                        r_state,
                        r_action,
                        r_done,
                        r_reward,
                        r_new_state,
                        r_target_goal,
                        r_target_gnet_state,
                        r_c_goal,
                        r_n_goal,
                    )
                self.action_replay_memory.update_trajectory_id()

            reward_history.append(total_reward)
            steps_history.append(steps)

            if not eval and episode % self.eval_every == 0 and episode > 0:
                self.action_model.eval()

                with torch.no_grad():
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
