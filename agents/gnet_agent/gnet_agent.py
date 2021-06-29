from agents.base.agent import Agent
from agents.gnet_agent.model import GNetGoalDQN

import torch
from tqdm import tqdm
import numpy as np
import pickle
import torch.optim as optim
from skimage.color import rgb2gray


class GNetAgent(Agent):
    def __init__(
        self,
        env,
        gnet_manager,
        action_model,
        goal_model,
        action_memory,
        goal_memory,
        action_space,
        goal_space,
        num_goals,
        idx2goalnet,
        use_ga,
        env_name,
        device,
    ):
        super(GNetAgent, self).__init__(
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
        self.use_ga = use_ga

        if self.env_name == "two_keys" or self.env_name == "four_rooms_3d":
            if self.env_name == "two_keys":
                self.goal_clone_interval = 10
            elif self.env_name == "four_rooms_3d":
                self.goal_clone_interval = 50
            self.goal_replay_memory = goal_memory
            self.goal_model = goal_model
            self.goal_tau = 1.0
            self.goal_lr = 0.001
            self.eps_hi = 0.5
            self.goal_opt = optim.Adam(self.goal_model.parameters(), lr=self.goal_lr)
            self.clone_goal_model = (
                GNetGoalDQN(num_goals, self.device, env_name).eval().to(self.device)
            )
            self.clone(self.clone_goal_model, self.goal_model, 1)

    def select_action(self, state, goal, c_goal, actions=None):
        state = torch.as_tensor(state, device=self.device).float().unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0)
        c_goal = torch.as_tensor(c_goal, device=self.device).float().unsqueeze(0)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device).float().unsqueeze(0)
        q_values = self.action_model(state, goal, c_goal, actions)
        action = torch.argmax(q_values)

        return action.item()

    def select_subgoal(self, state, goalnet_code, mask, state_extra=None):
        state = torch.as_tensor(state, device=self.device).float().unsqueeze(0)
        goalnet_code = torch.as_tensor(goalnet_code, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        if state_extra is not None:
            state_extra = torch.as_tensor(state_extra, device=self.device).float()
        q_values = self.goal_model(state, goalnet_code, mask, state_extra)
        subgoal = torch.argmax(q_values)

        subgoal_coords = self.gnet_manager.get_goal_state(self.idx2goalnet[subgoal])
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
                c_goal,
                n_goal,
            ) = self.action_replay_memory.retrieve()
            q = (
                self.action_model(state, goal, c_goal)
                .gather(1, action.view(self.minibatch_size, 1))
                .squeeze(1)
            )
            a_max = self.action_model(next_state, goal, n_goal).max(dim=1)[1].detach()
            qmax = (
                self.clone_action_model(next_state, goal, n_goal)
                .detach()
                .gather(1, a_max.view(self.minibatch_size, 1))
                .squeeze(1)
            )
        elif self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            (
                reward,
                terminal,
                goal,
                indexes,
            ) = self.action_replay_memory.retrieve_frame_stack()
            (
                state,
                action,
                c_goal,
                next_state,
                n_goal,
            ) = self.action_replay_memory.get_frame_stack(indexes)
            q = (
                self.action_model(state, goal, c_goal, action[:, :-1])
                .squeeze(1)
                .gather(1, action[:, -1].view(self.minibatch_size, 1))
                .squeeze(1)
            )
            a_max = (
                self.action_model(next_state, goal, n_goal, action[:, 1:])
                .squeeze(1)
                .max(dim=1)[1]
                .detach()
            )
            qmax = (
                self.clone_action_model(next_state, goal, n_goal, action[:, 1:])
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
        torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), 1.0)
        self.action_opt.step()

    def update_goal_model(self):
        if self.env_name == "two_keys":
            (
                state,
                action,
                reward,
                terminal,
                next_state,
                gnet_state,
                gnet_mask,
                next_gnet_state,
                next_gnet_mask,
                steps,
                _,
                _,
            ) = self.goal_replay_memory.retrieve()
            q = (
                self.goal_model(state, gnet_state, gnet_mask)
                .gather(1, action.view(self.minibatch_size, 1))
                .squeeze(1)
            )
            g_max = (
                self.goal_model(next_state, next_gnet_state, next_gnet_mask)
                .max(dim=1)[1]
                .detach()
            )
            qmax = (
                self.clone_goal_model(next_state, next_gnet_state, next_gnet_mask)
                .detach()
                .gather(1, g_max.view(self.minibatch_size, 1))
                .squeeze(1)
            )
        elif self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            (
                state,
                action,
                reward,
                terminal,
                next_state,
                gnet_state,
                gnet_mask,
                next_gnet_state,
                next_gnet_mask,
                steps,
                state_extra,
                next_state_extra,
            ) = self.goal_replay_memory.retrieve()
            q = (
                self.goal_model(state, gnet_state, gnet_mask, state_extra)
                .gather(1, action.view(self.minibatch_size, 1))
                .squeeze(1)
            )
            g_max = (
                self.goal_model(
                    next_state, next_gnet_state, next_gnet_mask, next_state_extra
                )
                .max(dim=1)[1]
                .detach()
            )
            qmax = (
                self.clone_goal_model(
                    next_state, next_gnet_state, next_gnet_mask, next_state_extra
                )
                .detach()
                .gather(1, g_max.view(self.minibatch_size, 1))
                .squeeze(1)
            )

        nonterminal_target = (self.gamma ** steps) * reward + (
            self.gamma ** steps
        ) * qmax
        terminal_target = (self.gamma ** steps) * reward

        target = (
            terminal.float() * terminal_target
            + (~terminal).float() * nonterminal_target
        )
        loss = self.loss(q, target)

        self.goal_opt.zero_grad()
        loss.backward()
        self.goal_opt.step()

    def save(self, result_data):
        super(GNetAgent, self).save(result_data)
        if self.env_name != "ai2thor_kitchen":
            torch.save(self.goal_model.state_dict(), self.save_dir + "/high_level.pt")
            torch.save(
                self.clone_goal_model.state_dict(),
                self.save_dir + "/high_level_clone.pt",
            )
            torch.save(self.goal_opt.state_dict(), self.save_dir + "/high_level_opt.pt")
            self.goal_replay_memory.save_to_disk()

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
        result_data = dict(episode=[], reward=[], steps=[], frames=[])

        reward_history = []
        steps_history = []

        subgoal_progress = {}
        for key in self.gnet_manager.gnet_goals.keys():
            subgoal_progress[key] = []

        if load:
            self.load()

        progress_bar = tqdm(range(episode, episodes), unit="episode", disable=eval)
        goal_steps = 0

        if self.env_name == "four_rooms_3d":
            channels = 3
            img_height = 60
            img_width = 80
            if self.use_ga:
                state_extra_size = 11
            else:
                state_extra_size = 8
        elif self.env_name == "ai2thor_kitchen":
            channels = 4
            img_height = 100
            img_width = 100
            if self.use_ga:
                state_extra_size = 10
            else:
                state_extra_size = 8

        for episode in progress_bar:
            total_steps = 0
            if self.env_name == "ai2thor_kitchen":
                obs = self.env.reset(train=train, idx=floor_idx, seed=seed)
            else:
                obs = self.env.reset()
            state = self.process_obs(obs)
            if self.env_name == "two_keys":
                obs = state
            self.gnet_manager.reset()
            state_goal_selection = state

            trajectory_data = []
            her_trajectory_data = []

            episode_done = False
            total_reward = 0

            c_goal_stack = None
            state_stack = None
            action_stack = None

            if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
                c_goal_stack = np.zeros((state_extra_size * self.frame_stack))
                state_stack = np.zeros(
                    (channels * self.frame_stack, img_height, img_width)
                )
                action_stack = np.zeros((self.frame_stack - 1))

            # Outer loop for high level
            while not episode_done:
                extrinsic_reward = 0
                if self.env_name == "four_rooms_3d":
                    hi_c_goal = [
                        self.env.agent.pos[0],
                        self.env.agent.pos[2],
                        self.env.goal.pos[0],
                        self.env.goal.pos[2],
                        self.env.yellow_subgoal.pos[0],
                        self.env.yellow_subgoal.pos[2],
                        self.env.blue_subgoal.pos[0],
                        self.env.blue_subgoal.pos[2],
                    ]
                else:
                    hi_c_goal = None
                if (
                    train
                    and self.env_name != "ai2thor_kitchen"
                    and np.random.uniform() < self.eps_hi
                ):
                    goalnet_code, mask = self.gnet_manager.generate_goal_mask()
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
                    subgoal_coord = self.gnet_manager.get_goal_state(chosen)

                    self.gnet_manager.gnet_state = self.gnet_manager.get_parent_goal(
                        chosen
                    )
                elif self.env_name == "ai2thor_kitchen":
                    chosen = self.gnet_manager.generate_options_list()[0][-1]
                    self.gnet_manager.gnet_state = self.gnet_manager.get_parent_goal(
                        chosen
                    )
                    subgoal_id = self.idx2goalnet.index(chosen)
                    subgoal_coord = self.gnet_manager.get_goal_state(chosen)
                else:
                    goalnet_code, mask = self.gnet_manager.generate_goal_mask()
                    subgoal_id, subgoal_coord = self.select_subgoal(
                        state, goalnet_code, mask, hi_c_goal
                    )

                    self.gnet_manager.gnet_state = self.gnet_manager.get_parent_goal(
                        self.idx2goalnet[subgoal_id]
                    )

                goal_data = []
                subgoal_data = []
                subgoal_achieved = False
                other_subgoal_achieved = False
                achieved_subgoal_id = 0

                # Inner loop for low level
                while (
                    not subgoal_achieved
                    and not episode_done
                    and not other_subgoal_achieved
                ):
                    c_goal = self.gnet_manager.current_goal_state()
                    if self.env_name == "four_rooms_3d":
                        if self.use_ga:
                            c_goal = [
                                self.env.goal.pos[0],
                                self.env.goal.pos[2],
                                self.env.yellow_subgoal.pos[0],
                                self.env.yellow_subgoal.pos[2],
                                self.env.blue_subgoal.pos[0],
                                self.env.blue_subgoal.pos[2],
                            ] + c_goal
                        else:
                            c_goal = [
                                self.env.agent.pos[0],
                                self.env.agent.pos[2],
                                self.env.goal.pos[0],
                                self.env.goal.pos[2],
                                self.env.yellow_subgoal.pos[0],
                                self.env.yellow_subgoal.pos[2],
                                self.env.blue_subgoal.pos[0],
                                self.env.blue_subgoal.pos[2],
                            ]

                        state_stack = np.concatenate((state_stack[channels:], state))
                        c_goal_stack = np.concatenate(
                            (c_goal_stack[state_extra_size:], c_goal)
                        )
                    elif self.env_name == "ai2thor_kitchen":
                        if self.use_ga:
                            c_goal = self.env.fridge_pos + self.env.light_pos + c_goal
                        else:
                            c_goal = (
                                [
                                    self.env.last_meta["agent"]["position"]["x"],
                                    self.env.last_meta["agent"]["position"]["z"],
                                ]
                                + self.env.fridge_pos
                                + self.env.light_pos
                            )

                        state_stack = np.concatenate((state_stack[channels:], state))
                        c_goal_stack = np.concatenate(
                            (c_goal_stack[state_extra_size:], c_goal)
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
                            action = self.select_action(state, subgoal_coord, c_goal)
                        else:
                            action = self.select_action(
                                state_stack, subgoal_coord, c_goal_stack, action_stack
                            )

                    if (
                        self.env_name == "four_rooms_3d"
                        or self.env_name == "ai2thor_kitchen"
                    ):
                        action_stack = np.append(action_stack[1:], action)

                    new_obs, reward, episode_done, info = self.env.step(action)

                    new_state = self.process_obs(new_obs)
                    n_goal = self.gnet_manager.current_goal_state()

                    if self.env_name == "two_keys":
                        new_obs = new_state

                    if self.env_name == "four_rooms_3d":
                        if self.use_ga:
                            n_goal = [
                                self.env.goal.pos[0],
                                self.env.goal.pos[2],
                                self.env.yellow_subgoal.pos[0],
                                self.env.yellow_subgoal.pos[2],
                                self.env.blue_subgoal.pos[0],
                                self.env.blue_subgoal.pos[2],
                            ] + n_goal
                        else:
                            n_goal = [
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
                        if self.use_ga:
                            n_goal = self.env.fridge_pos + self.env.light_pos + n_goal
                        else:
                            n_goal = (
                                [
                                    self.env.last_meta["agent"]["position"]["x"],
                                    self.env.last_meta["agent"]["position"]["z"],
                                ]
                                + self.env.fridge_pos
                                + self.env.light_pos
                            )

                    total_steps += 1

                    if self.gnet_manager.check_goal_satisfied(
                        self.idx2goalnet[subgoal_id]
                    ):
                        subgoal_achieved = True
                        subgoal_reward = 1
                    else:
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

                        subgoal_data.append(
                            (
                                obs,
                                action,
                                done,
                                subgoal_reward,
                                new_obs,
                                self.gnet_manager.target_goal_state,
                                c_goal,
                                n_goal,
                            )
                        )
                        goal_data.append((obs, episode_done))
                        if self.env_name == "two_keys":
                            self.action_replay_memory.save(
                                obs,
                                action,
                                done,
                                subgoal_reward,
                                new_obs,
                                self.gnet_manager.target_goal_state,
                                c_goal,
                                n_goal,
                            )
                        else:
                            trajectory_data.append(
                                (
                                    obs,
                                    action,
                                    done,
                                    subgoal_reward,
                                    new_obs,
                                    self.gnet_manager.target_goal_state,
                                    c_goal,
                                    n_goal,
                                )
                            )

                    state = new_state
                    obs = new_obs
                    total_reward += reward
                    extrinsic_reward += reward

                    if not eval:
                        self.frame += 1

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
                    add_experience = False
                    new_goal = self.gnet_manager.current_goal_state()
                    first = True
                    if self.env_name == "two_keys":
                        for (
                            r_state,
                            r_action,
                            r_done,
                            _,
                            r_new_state,
                            _,
                            c_goal,
                            n_goal,
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
                                c_goal,
                                n_goal,
                            )
                    else:
                        (
                            old_state,
                            old_action,
                            old_done,
                            old_reward,
                            old_new_state,
                            old_goal,
                            old_c_goal,
                            old_n_goal,
                        ) = subgoal_data[-1]
                        subgoal_data[-1] = (
                            old_state,
                            old_action,
                            old_done,
                            1,
                            old_new_state,
                            old_goal,
                            old_c_goal,
                            old_n_goal,
                        )
                        for (
                            r_state,
                            r_action,
                            r_done,
                            r_reward,
                            r_new_state,
                            _,
                            r_c_goal,
                            r_n_goal,
                        ) in subgoal_data:
                            her_trajectory_data.append(
                                (
                                    r_state,
                                    r_action,
                                    r_done,
                                    r_reward,
                                    r_new_state,
                                    new_goal,
                                    r_c_goal,
                                    r_n_goal,
                                )
                            )
                    self.gnet_manager.update_success_rate(
                        self.idx2goalnet[subgoal_id], subgoal_achieved
                    )
                    goal_steps += 1

                    if subgoal_achieved:
                        subgoal_progress[self.idx2goalnet[subgoal_id]].append(1)
                        add_experience = True
                    # Add a relabelled transition for the goal selection model
                    elif (
                        self.env_name == "two_keys"
                        and self.env.agent_pos.tolist() == self.env.goal_pos.tolist()
                    ):
                        add_experience = True
                    elif self.env_name == "four_rooms_3d" and self.env.near(
                        self.env.goal
                    ):
                        add_experience = True
                    elif other_subgoal_achieved:
                        subgoal_progress[self.idx2goalnet[subgoal_id]].append(0)
                        add_experience = True
                    else:
                        subgoal_progress[self.idx2goalnet[subgoal_id]].append(0)
                    if add_experience:
                        steps = 1
                        next_gnet_state = self.idx2goalnet[r_subgoal_id]
                        self.gnet_manager.set_state(r_gnet_goal, next_gnet_state)
                        if self.env_name != "ai2thor_kitchen":
                            (
                                next_goalnet_code,
                                next_mask,
                            ) = self.gnet_manager.generate_goal_mask()

                            if self.env_name == "four_rooms_3d":
                                hi_n_goal = [
                                    self.env.agent.pos[0],
                                    self.env.agent.pos[2],
                                    self.env.goal.pos[0],
                                    self.env.goal.pos[2],
                                    self.env.yellow_subgoal.pos[0],
                                    self.env.yellow_subgoal.pos[2],
                                    self.env.blue_subgoal.pos[0],
                                    self.env.blue_subgoal.pos[2],
                                ]
                            for r_state, r_done in goal_data[::-1]:
                                if self.env_name == "two_keys":
                                    self.goal_replay_memory.save(
                                        r_state,
                                        r_subgoal_id,
                                        episode_done,
                                        extrinsic_reward,
                                        new_state,
                                        goalnet_code,
                                        self.gnet_manager.gnet_goals[next_gnet_state][
                                            "code"
                                        ],
                                        mask,
                                        next_mask,
                                        steps,
                                    )
                                elif self.env_name == "two_keys":
                                    self.goal_replay_memory.save(
                                        r_state,
                                        r_subgoal_id,
                                        episode_done,
                                        extrinsic_reward,
                                        new_obs,
                                        goalnet_code,
                                        gnet_goals[next_gnet_state]["code"],
                                        mask,
                                        next_mask,
                                        steps,
                                        hi_c_goal,
                                        hi_n_goal,
                                    )
                                steps += 1
                    if self.env_name != "ai2thor_kitchen":
                        if self.goal_replay_memory.current_size > self.minibatch_size:
                            self.update_goal_model()

                        if goal_steps % self.goal_clone_interval == 0:
                            self.clone(
                                self.clone_goal_model, self.goal_model, self.goal_tau
                            )
                elif (subgoal_achieved or other_subgoal_achieved) and not episode_done:
                    self.gnet_manager.set_state(
                        r_gnet_goal, self.idx2goalnet[r_subgoal_id]
                    )
                state_goal_selection = new_state

            if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
                for (
                    r_state,
                    r_action,
                    r_done,
                    r_reward,
                    r_new_state,
                    r_target_goal,
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
                        r_c_goal,
                        r_n_goal,
                    )
                self.action_replay_memory.update_trajectory_id()

            reward_history.append(total_reward)
            steps_history.append(total_steps)

            if not eval and episode % self.eval_every == 0 and episode > 0:
                if self.env_name != "ai2thor_kitchen":
                    self.goal_model.eval()
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
                if self.env_name != "ai2thor_kitchen":
                    self.goal_model.train()
                self.action_model.train()

                if do_print:
                    self.print_stats(
                        reward_history, steps_history, result_data, subgoal_progress
                    )
            elif eval:
                result_data["episode"].append(episode)
                result_data["reward"].append(total_reward)
                result_data["steps"].append(total_steps)

            if save_checkpoints and (
                episode % self.save_every == 0 and episode > 0 and not eval
            ):
                self.save(result_data)

        return result_data
