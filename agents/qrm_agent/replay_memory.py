from agents.base.replay_memory import ReplayMemory
import numpy as np
import random
import torch


class QrmReplayMemory(ReplayMemory):
    def __init__(self, size, num_goals, device, minibatch_size, env_name, load=False):
        super(QrmReplayMemory, self).__init__(
            size, device, minibatch_size, env_name, load
        )
        self.size = size
        self.minibatch_size = minibatch_size
        self.gnet_states = np.empty(self.size, dtype=(np.int32, num_goals))
        self.next_gnet_states = np.empty(self.size, dtype=(np.int32, num_goals))

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            self.state_extra = np.empty(self.size, dtype=(np.float, 8))
            self.next_state_extra = np.empty(self.size, dtype=(np.float, 8))

        if load:
            self.load_from_disk()

    def load_from_disk(self):
        super(QrmReplayMemory, self).load_from_disk()

        infile = open(self.save_dir + "/gnet_states.npy", "rb")
        self.gnet_states = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/next_gnet_states.npy", "rb")
        self.next_gnet_states = np.load(infile)
        infile.close()

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            infile = open(self.save_dir + "/state_extra.npy", "rb")
            self.state_extra = np.load(infile)
            infile.close()

            infile = open(self.save_dir + "/next_state_extra.npy", "rb")
            self.next_state_extra = np.load(infile)
            infile.close()

    def save_to_disk(self):
        super(QrmReplayMemory, self).save_to_disk()

        outfile = open(self.save_dir + "/gnet_states.npy", "wb")
        np.save(outfile, self.gnet_states)
        outfile.close()

        outfile = open(self.save_dir + "/next_gnet_states.npy", "wb")
        np.save(outfile, self.next_gnet_states)
        outfile.close()

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            outfile = open(self.save_dir + "/state_extra.npy", "wb")
            np.save(outfile, self.state_extra)
            outfile.close()

            outfile = open(self.save_dir + "/next_state_extra.npy", "wb")
            np.save(outfile, self.next_state_extra)
            outfile.close()

    def save(
        self,
        state,
        action,
        done,
        reward,
        next_state,
        gnet_state,
        next_gnet_state,
        state_extra=None,
        next_state_extra=None,
    ):
        super(QrmReplayMemory, self).save(state, action, done, reward, next_state)
        self.gnet_states[self.current_index] = gnet_state
        self.next_gnet_states[self.current_index] = next_gnet_state

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            self.state_extra[self.current_index] = state_extra
            self.next_state_extra[self.current_index] = next_state_extra

        self.current_size = max(self.current_size, self.current_index + 1)
        self.current_index = (self.current_index + 1) % self.size

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
            torch.as_tensor(self.gnet_states[indexes], device=self.device).long(),
            torch.as_tensor(self.next_gnet_states[indexes], device=self.device).long(),
        )

    # Retrieve function for environments that use frame stacking
    def retrieve_frame_stack(self):
        indexes = [
            np.random.randint(self.current_size) for i in range(self.minibatch_size)
        ]

        return (
            torch.as_tensor(self.rewards[indexes], device=self.device),
            torch.as_tensor(self.done_flags[indexes], device=self.device),
            torch.as_tensor(self.gnet_states[indexes], device=self.device).long(),
            torch.as_tensor(self.next_gnet_states[indexes], device=self.device).long(),
            indexes,
        )
