from gnet.gnet_manager import GNetManager
import numpy as np


class KitchenGNetManager(GNetManager):
    def __init__(self, env, gnet_goals, max_exploration_rate=0.8):
        super().__init__(env, gnet_goals, max_exploration_rate=max_exploration_rate)

        # agent x/y, fridge open, light on
        self.target_goal_state = [0, 0, 1, 1]

        self.init_goal_state = [0, 0, 1, 1]

        self.gnet_goals = gnet_goals

        self.fridge_closed = False
        self.microwave_closed = False

        self.action_space_size = 5
        self.goal_space_size = 4

    def get_goal_state(self, selected_goal):
        self.target_goal_state = self.current_goal_state()

        if selected_goal == "fridge closed":
            self.target_goal_state[0] = self.env.fridge_pos[0]
            self.target_goal_state[1] = self.env.fridge_pos[2]
            self.target_goal_state[2] = 0
        elif selected_goal == "end":
            self.target_goal_state[0] = self.env.light_pos[0]
            self.target_goal_state[1] = self.env.light_pos[2]
            self.target_goal_state[3] = 0

        return self.target_goal_state

    def check_goal_satisfied(self, selected_goal):
        satisfied = False
        goal_state = self.current_goal_state()

        if selected_goal == "fridge closed" and not self.fridge_closed:
            satisfied = goal_state[2] == 0
            if satisfied:
                self.fridge_closed = True
        elif selected_goal == "end":
            satisfied = goal_state[3] == 0

        return satisfied

    def current_goal_state(self):
        fridge = None
        light = None

        agent_x = self.env.last_meta["agent"]["position"]["x"]
        agent_y = self.env.last_meta["agent"]["position"]["z"]

        for i in self.env.last_meta["objects"]:
            if i["name"].find("Fridge") == 0:
                fridge = i
            elif i["name"].find("LightSwitch") == 0:
                light = i

        goal = [agent_x, agent_y, int(fridge["isOpen"]), int(light["isToggled"])]

        return goal

    def reset(self):
        super().reset()
        self.init_goal_state = self.current_goal_state()
        self.fridge_closed = False
