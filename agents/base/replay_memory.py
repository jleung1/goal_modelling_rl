import numpy as np
import pickle
import torch


class ReplayMemory:
    def __init__(
        self,
        size,
        device,
        minibatch_size,
        env_name,
        load=False,
        save_dir="./save/replay",
    ):
        self.size = size
        self.minibatch_size = minibatch_size
        self.device = device
        self.env_name = env_name
        self.gnet_model = False

        if self.env_name == "two_keys":
            self.states = np.empty((self.size, 3, 13, 13), dtype=np.float32)
            self.next_states = np.empty((self.size, 3, 13, 13), dtype=np.float32)
        elif self.env_name == "four_rooms_3d":
            self.states = np.empty((self.size, 60, 80, 3), dtype=np.uint8)
            self.next_states = np.empty((self.size, 60, 80, 3), dtype=np.uint8)
            self.trajectory_ids = np.empty(self.size, dtype=np.int32)
            self.trajectory_id = 0
            self.max_trajectory_id = size
        elif self.env_name == "ai2thor_kitchen":
            self.states = np.empty((self.size, 100, 100, 4), dtype=np.uint8)
            self.next_states = np.empty((self.size, 100, 100, 4), dtype=np.uint8)
            self.trajectory_ids = np.empty(self.size, dtype=np.int32)
            self.trajectory_id = 0
            self.max_trajectory_id = size
        self.actions = np.empty(self.size, dtype=np.int32)
        self.done_flags = np.empty(self.size, dtype=np.bool)
        self.rewards = np.empty(self.size, dtype=np.float32)

        self.current_index = 0
        self.current_size = 0

        self.frame_stack = 4

        self.save_dir = save_dir

        if load:
            self.load_from_disk()

    def save(self, state, action, done, reward, next_state):
        self.actions[self.current_index] = action
        self.states[self.current_index, ...] = state
        self.next_states[self.current_index, ...] = next_state
        self.rewards[self.current_index] = reward
        self.done_flags[self.current_index] = done

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            self.trajectory_ids[self.current_index] = self.trajectory_id

    def load_from_disk(self):
        infile = open(self.save_dir + "/states.npy", "rb")
        self.states = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/next_states.npy", "rb")
        self.next_states = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/actions.npy", "rb")
        self.actions = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/done_flags.npy", "rb")
        self.done_flags = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/rewards.npy", "rb")
        self.rewards = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/replay_info.npy", "rb")
        info = pickle.load(infile)
        infile.close()

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            infile = open(self.save_dir + "/trajectory_ids.npy", "rb")
            self.trajectory_ids = np.load(infile)
            infile.close()

        self.current_index = info["current_index"]
        self.current_size = info["current_size"]

    def save_to_disk(self):
        outfile = open(self.save_dir + "/states.npy", "wb")
        np.save(outfile, self.states)
        outfile.close()

        outfile = open(self.save_dir + "/next_states.npy", "wb")
        np.save(outfile, self.next_states)
        outfile.close()

        outfile = open(self.save_dir + "/actions.npy", "wb")
        np.save(outfile, self.actions)
        outfile.close()

        outfile = open(self.save_dir + "/done_flags.npy", "wb")
        np.save(outfile, self.done_flags)
        outfile.close()

        outfile = open(self.save_dir + "/rewards.npy", "wb")
        np.save(outfile, self.rewards)
        outfile.close()

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            outfile = open(self.save_dir + "/trajectory_ids.npy", "wb")
            np.save(outfile, self.trajectory_ids)
            outfile.close()

        outfile = open(self.save_dir + "/replay_info.npy", "wb")
        pickle.dump(
            {"current_index": self.current_index, "current_size": self.current_size},
            outfile,
        )
        outfile.close()

    def retrieve(self):
        indexes = [
            random.randint(0, self.current_size - 1) for i in range(self.minibatch_size)
        ]

        return (
            torch.as_tensor(self.states[indexes], device=self.device).float(),
            torch.as_tensor(self.actions[indexes], device=self.device).long(),
            torch.as_tensor(self.rewards[indexes], device=self.device),
            torch.as_tensor(self.done_flags[indexes], device=self.device),
            torch.as_tensor(self.next_states[indexes], device=self.device).float(),
        )

    def update_trajectory_id(self):
        self.trajectory_id = (self.trajectory_id + 1) % self.max_trajectory_id

    # Get the state/goal that is n steps from the ref index for environments that use frame stacking
    def get_frame_stack(self, ref_indexes, extra_size=8):
        if self.env_name == "four_rooms_3d":
            channels = 3
            img_height = 60
            img_width = 80
        elif self.env_name == "ai2thor_kitchen":
            channels = 4
            img_height = 100
            img_width = 100
        ret_states = torch.zeros(
            (self.minibatch_size, channels * self.frame_stack, img_height, img_width),
            device=self.device,
        )
        ret_next_states = torch.zeros(
            (self.minibatch_size, channels * self.frame_stack, img_height, img_width),
            device=self.device,
        )
        ret_state_extra = torch.zeros(
            (self.minibatch_size, extra_size * self.frame_stack), device=self.device
        )
        ret_next_state_extra = torch.zeros(
            (self.minibatch_size, extra_size * self.frame_stack), device=self.device
        )
        ret_actions = torch.zeros(
            (self.minibatch_size, self.frame_stack), device=self.device
        )

        for step in range(1, self.frame_stack + 1):
            indexes = np.array(ref_indexes) - self.frame_stack + step
            negative = indexes < 0
            indexes[negative] = self.size + indexes[negative]

            states = (
                torch.as_tensor(self.states[indexes], device=self.device)
                .float()
                .permute(0, 3, 1, 2)
                / 255.0
            ).contiguous()
            next_states = (
                torch.as_tensor(self.next_states[indexes], device=self.device)
                .float()
                .permute(0, 3, 1, 2)
                / 255.0
            ).contiguous()
            actions = torch.as_tensor(self.actions[indexes], device=self.device)

            # Names are different for GNet and GNet w/o GA
            if self.gnet_model:
                state_extra = torch.as_tensor(
                    self.cur_goal_s[indexes], device=self.device
                ).float()
                next_state_extra = torch.as_tensor(
                    self.next_goal_s[indexes], device=self.device
                ).float()
            else:
                state_extra = torch.as_tensor(
                    self.state_extra[indexes], device=self.device
                ).float()
                next_state_extra = torch.as_tensor(
                    self.next_state_extra[indexes], device=self.device
                ).float()

            start_traj_ids = torch.as_tensor(
                self.trajectory_ids[ref_indexes], device=self.device
            )
            traj_ids = torch.as_tensor(self.trajectory_ids[indexes], device=self.device)

            same_ids = start_traj_ids != traj_ids

            states[same_ids] = 0
            state_extra[same_ids] = 0
            next_states[same_ids] = 0
            next_state_extra[same_ids] = 0
            actions[same_ids] = 0

            ret_states[:, channels * (step - 1) : channels * step] = states
            ret_next_states[:, channels * (step - 1) : channels * step] = next_states
            ret_state_extra[
                :, extra_size * (step - 1) : extra_size * step
            ] = state_extra
            ret_next_state_extra[
                :, extra_size * (step - 1) : extra_size * step
            ] = next_state_extra

            ret_actions[:, step - 1] = actions
        return (
            ret_states,
            ret_actions.long(),
            ret_state_extra,
            ret_next_states,
            ret_next_state_extra,
        )
