from gnet.gnet_manager import GNetManager
import numpy as np

DEBUG = False


class FourRoomSubgoals3DGNetManager(GNetManager):
    def __init__(self, env, gnet_goals, qrm=False):
        super().__init__(env, gnet_goals, last_num_goals=50, max_exploration_rate=0.8)
        # Goal space is [yellow subgoal visited, blue subgoal visited, goal reached]
        self.target_goal_state = [0, 0, 0, 0, 0]

        self.gnet_goals = gnet_goals
        self.qrm = qrm

        self.reached_yellow_subgoal = False
        self.reached_blue_subgoal = False

        self.goal_space_size = 5
        self.action_space_size = 3

    def get_goal_state(self, selected_goal):
        self.target_goal_state = self.current_goal_state()
        if selected_goal == "reached yellow subgoal" or (
            self.qrm
            and selected_goal == "reached all subgoals"
            and not self.reached_yellow_subgoal
        ):
            x = self.env.yellow_subgoal.pos[0]
            y = self.env.yellow_subgoal.pos[2]
            self.target_goal_state[2] = True
        elif selected_goal == "reached blue subgoal" or (
            self.qrm
            and selected_goal == "reached all subgoals"
            and not self.reached_blue_subgoal
        ):
            x = self.env.blue_subgoal.pos[0]
            y = self.env.blue_subgoal.pos[2]
            self.target_goal_state[3] = True
        elif selected_goal == "end":
            x = self.env.goal.pos[0]
            y = self.env.goal.pos[2]
            self.target_goal_state[4] = True

        self.target_goal_state[0] = x
        self.target_goal_state[1] = y

        return self.target_goal_state

    def check_goal_satisfied(self, selected_goal):
        if (
            selected_goal == "reached yellow subgoal"
            and not self.reached_yellow_subgoal
            or (
                self.qrm
                and selected_goal == "reached all subgoals"
                and not self.reached_yellow_subgoal
            )
        ):
            self.reached_yellow_subgoal = self.env.reached_yellow_subgoal
            if self.reached_yellow_subgoal:
                if DEBUG:
                    print("reached yellow subgoal!")
                return True

        elif (
            selected_goal == "reached blue subgoal"
            and not self.reached_blue_subgoal
            or (
                self.qrm
                and selected_goal == "reached all subgoals"
                and not self.reached_blue_subgoal
            )
        ):
            self.reached_blue_subgoal = self.env.reached_blue_subgoal
            if self.reached_blue_subgoal:
                if DEBUG:
                    print("reached blue subgoal!")
                return True

        elif selected_goal == "end":
            satisfied = self.env.near(self.env.goal)
            if satisfied:
                if DEBUG:
                    print("goal reached!")
                return True

        return False

    def check_goal_relabel(self, start_goal, selected_goal):
        satisfied = False
        if selected_goal == "reached yellow subgoal" or (
            self.qrm
            and start_goal == "reached blue subgoal"
            and selected_goal == "reached all subgoals"
        ):
            satisfied = self.env.near(self.env.yellow_subgoal)

        elif selected_goal == "reached blue subgoal" or (
            self.qrm
            and start_goal == "reached yellow subgoal"
            and selected_goal == "reached all subgoals"
        ):
            satisfied = self.env.near(self.env.blue_subgoal)

        elif selected_goal == "end":
            satisfied = self.env.near(self.env.goal)

        return satisfied

    def current_goal_state(self):
        x = self.env.agent.pos[0]
        y = self.env.agent.pos[2]

        return [
            x,
            y,
            self.reached_yellow_subgoal,
            self.reached_blue_subgoal,
            self.env.near(self.env.goal),
        ]

    def reset(self):
        super().reset()
        self.reached_yellow_subgoal = False
        self.reached_blue_subgoal = False

        self.target_goal_state = [0, 0, False, False, False]
