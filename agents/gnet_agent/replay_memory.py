from agents.base.replay_memory import ReplayMemory
import numpy as np
import random
import torch


class GNetActionReplayMemory(ReplayMemory):
    def __init__(
        self,
        size,
        goal_space_size,
        device,
        minibatch_size,
        env_name,
        use_ga,
        load=False,
    ):
        super(GNetActionReplayMemory, self).__init__(
            size, device, minibatch_size, env_name, load, "./save/low_replay"
        )

        self.goals = np.empty(self.size, dtype=(np.int32, goal_space_size))
        self.use_ga = use_ga
        self.gnet_model = True
        if self.env_name == "two_keys":
            self.cur_goal_s = np.empty(self.size, dtype=(np.int32, goal_space_size))
            self.next_goal_s = np.empty(self.size, dtype=(np.int32, goal_space_size))
        elif self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            if use_ga:
                size = goal_space_size + 6
            else:
                size = 8
            self.cur_goal_s = np.empty(self.size, dtype=(np.float32, size))
            self.next_goal_s = np.empty(self.size, dtype=(np.float32, size))

        if load:
            self.load_from_disk()

    def save(self, state, action, done, reward, next_state, goal, c_goal, n_goal):
        super(GNetActionReplayMemory, self).save(
            state, action, done, reward, next_state
        )
        self.goals[self.current_index] = goal
        self.cur_goal_s[self.current_index] = c_goal
        self.next_goal_s[self.current_index] = n_goal

        self.current_size = max(self.current_size, self.current_index + 1)
        self.current_index = (self.current_index + 1) % self.size

    def load_from_disk(self):
        super(GNetActionReplayMemory, self).load_from_disk()

        infile = open(self.save_dir + "/goals.npy", "rb")
        self.goals = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/cur_goal_s.npy", "rb")
        self.cur_goal_s = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/next_goal_s.npy", "rb")
        self.next_goal_s = np.load(infile)
        infile.close()

    def save_to_disk(self):
        super(GNetActionReplayMemory, self).save_to_disk()

        outfile = open(self.save_dir + "/goals.npy", "wb")
        np.save(outfile, self.goals)
        outfile.close()

        outfile = open(self.save_dir + "/cur_goal_s.npy", "wb")
        np.save(outfile, self.cur_goal_s)
        outfile.close()

        outfile = open(self.save_dir + "/next_goal_s.npy", "wb")
        np.save(outfile, self.next_goal_s)
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
            torch.as_tensor(self.goals[indexes], device=self.device).long(),
            torch.as_tensor(self.cur_goal_s[indexes], device=self.device).long(),
            torch.as_tensor(self.next_goal_s[indexes], device=self.device).long(),
        )

    def get_frame_stack(self, ref_indexes):
        if self.use_ga:
            if self.env_name == "four_rooms_3d":
                size = 11
            elif self.env_name == "ai2thor_kitchen":
                size = 10
        else:
            size = 8
        return super(GNetActionReplayMemory, self).get_frame_stack(ref_indexes, size)

    # Retrieve function for environments that use frame stacking
    def retrieve_frame_stack(self):
        indexes = [
            np.random.randint(self.current_size) for i in range(self.minibatch_size)
        ]

        return (
            torch.as_tensor(self.rewards[indexes], device=self.device),
            torch.as_tensor(self.done_flags[indexes], device=self.device),
            torch.as_tensor(self.goals[indexes], device=self.device).long(),
            indexes,
        )


class GNetGoalReplayMemory(ReplayMemory):
    def __init__(self, size, num_goals, device, minibatch_size, env_name, load=False):
        super(GNetGoalReplayMemory, self).__init__(
            size, device, minibatch_size, env_name, load, "./save/high_replay"
        )
        self.gnet_states = np.empty((self.size, num_goals), dtype=np.int32)
        self.gnet_masks = np.empty((self.size, num_goals), dtype=np.float32)
        self.next_gnet_states = np.empty((self.size, num_goals), dtype=np.int32)
        self.next_gnet_masks = np.empty((self.size, num_goals), dtype=np.float32)
        self.steps = np.empty(self.size, dtype=np.int32)

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            self.state_extra = np.empty(self.size, dtype=(np.float32, 8))
            self.next_state_extra = np.empty(self.size, dtype=(np.float32, 8))

        if load:
            self.load_from_disk()

    def save(
        self,
        state,
        action,
        done,
        reward,
        next_state,
        gnet_state,
        next_gnet_state,
        gnet_mask,
        next_gnet_mask,
        steps,
        state_extra=None,
        next_state_extra=None,
    ):
        super(GNetGoalReplayMemory, self).save(state, action, done, reward, next_state)

        self.gnet_states[self.current_index] = gnet_state
        self.gnet_masks[self.current_index] = gnet_mask
        self.next_gnet_states[self.current_index] = next_gnet_state
        self.next_gnet_masks[self.current_index] = next_gnet_mask
        self.steps[self.current_index] = steps

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            self.state_extra[self.current_index] = state_extra
            self.next_state_extra[self.current_index] = next_state_extra

        self.current_size = max(self.current_size, self.current_index + 1)
        self.current_index = (self.current_index + 1) % self.size

    def load_from_disk(self):
        super(GNetGoalReplayMemory, self).load_from_disk()
        infile = open(self.save_dir + "/gnet_states.npy", "rb")
        self.gnet_states = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/gnet_masks.npy", "rb")
        self.gnet_masks = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/next_gnet_states.npy", "rb")
        self.next_gnet_states = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/next_gnet_masks.npy", "rb")
        self.next_gnet_masks = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/steps.npy", "rb")
        self.steps = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/state_extra.npy", "rb")
        self.state_extra = np.load(infile)
        infile.close()

        infile = open(self.save_dir + "/next_state_extra.npy", "rb")
        self.next_state_extra = np.load(infile)
        infile.close()

    def save_to_disk(self):
        super(GNetGoalReplayMemory, self).save_to_disk()
        outfile = open(self.save_dir + "/gnet_states.npy", "wb")
        np.save(outfile, self.gnet_states)
        outfile.close()

        outfile = open(self.save_dir + "/gnet_masks.npy", "wb")
        np.save(outfile, self.gnet_masks)
        outfile.close()

        outfile = open(self.save_dir + "/next_gnet_states.npy", "wb")
        np.save(outfile, self.next_gnet_states)
        outfile.close()

        outfile = open(self.save_dir + "/next_gnet_masks.npy", "wb")
        np.save(outfile, self.next_gnet_masks)
        outfile.close()

        outfile = open(self.save_dir + "/steps.npy", "wb")
        np.save(outfile, self.steps)
        outfile.close()

        outfile = open(self.save_dir + "/state_extra.npy", "wb")
        np.save(outfile, self.state_extra)
        outfile.close()

        outfile = open(self.save_dir + "/next_state_extra.npy", "wb")
        np.save(outfile, self.next_state_extra)
        outfile.close()

    def retrieve(self):
        indexes = [
            np.random.randint(self.current_size) for i in range(self.minibatch_size)
        ]

        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            state_extra = torch.as_tensor(
                self.state_extra[indexes], device=self.device
            ).float()
            next_state_extra = torch.as_tensor(
                self.next_state_extra[indexes], device=self.device
            ).float()
        else:
            state_extra = None
            next_state_extra = None
        return (
            torch.as_tensor(self.states[indexes], device=self.device).float(),
            torch.as_tensor(self.actions[indexes], device=self.device).long(),
            torch.as_tensor(self.rewards[indexes], device=self.device),
            torch.as_tensor(self.done_flags[indexes], device=self.device),
            torch.as_tensor(self.next_states[indexes], device=self.device).float(),
            torch.as_tensor(self.gnet_states[indexes], device=self.device).long(),
            torch.as_tensor(self.gnet_masks[indexes], device=self.device).float(),
            torch.as_tensor(self.next_gnet_states[indexes], device=self.device).long(),
            torch.as_tensor(self.next_gnet_masks[indexes], device=self.device).float(),
            torch.as_tensor(self.steps[indexes], device=self.device).long(),
            state_extra,
            next_state_extra,
        )
