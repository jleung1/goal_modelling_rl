from gnet.gnet_manager import GNetManager
from env.two_keys import TwoKeysEnv
import numpy as np

DEBUG = False


class TwoKeysGNetManager(GNetManager):
    def __init__(self, env, gnet_goals):
        super().__init__(env, gnet_goals)
        # Goal space is [(sub) goal X, (sub) goal Y, have yellow key, have green key, yellow door open, green door open]
        self.target_goal_state = [0, 0, 0, 0, 0, 0]

        self.gnet_goals = gnet_goals

        self.goal_space_size = 6
        self.action_space_size = 5

    def get_goal_state(self, selected_goal):
        self.target_goal_state = self.current_goal_state()

        if selected_goal == "obtained yellow key":
            x = self.env.yellow_key_pos[0]
            y = self.env.yellow_key_pos[1]
            has_yellow_key = True
            self.target_goal_state[2] = has_yellow_key
        elif selected_goal == "obtained green key":
            x = self.env.green_key_pos[0]
            y = self.env.green_key_pos[1]
            has_green_key = True
            self.target_goal_state[3] = has_green_key
        elif selected_goal == "opened yellow door":
            x = self.env.yellow_door_pos[0]
            y = self.env.yellow_door_pos[1]
            self.target_goal_state[4] = True
        elif selected_goal == "opened green door":
            x = self.env.green_door_pos[0]
            y = self.env.green_door_pos[1]
            self.target_goal_state[5] = True
        elif selected_goal == "end":
            x = self.env.goal_pos[0]
            y = self.env.goal_pos[1]

        self.target_goal_state[0] = x
        self.target_goal_state[1] = y

        return self.target_goal_state

    def check_goal_satisfied(self, selected_goal):
        satisfied = False
        if selected_goal == "obtained yellow key":
            satisfied = self.env.carrying == self.env.yellow_key
            if DEBUG and satisfied:
                print("obtained yellow key!")

        elif selected_goal == "obtained green key":
            satisfied = self.env.carrying == self.env.green_key
            if DEBUG and satisfied:
                print("obtained green key!")

        elif selected_goal == "opened yellow door":
            satisfied = self.env.yellow_door.is_open
            if DEBUG and satisfied:
                print("yellow door opened!")

        elif selected_goal == "opened green door":
            satisfied = self.env.green_door.is_open
            if DEBUG and satisfied:
                print("green door opened!")

        elif selected_goal == "end":
            satisfied = self.env.agent_pos.tolist() == self.env.goal_pos.tolist()
            if DEBUG and satisfied:
                print("goal reached!")

        return satisfied

    def current_goal_state(self):
        x = self.env.agent_pos[0]
        y = self.env.agent_pos[1]
        has_yellow_key = self.env.carrying == self.env.yellow_key
        has_green_key = self.env.carrying == self.env.green_key
        yellow_door_open = self.env.yellow_door.is_open
        green_door_open = self.env.green_door.is_open

        return [x, y, has_yellow_key, has_green_key, yellow_door_open, green_door_open]
