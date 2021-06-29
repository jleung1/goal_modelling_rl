import numpy as np


class GoalNetGoal:
    def __init__(self, name, code, input_goals, output_goals):
        self.name = name
        self.code = code
        self.input_goals = input_goals
        self.output_goals = output_goals


class GoalNetPath:
    def __init__(self, current_gnet, options, waiting=0):
        self.current_gnet_goal = current_gnet
        self.options = options
        self.waiting = waiting
        self.wait_list = []

    def goal_in_options(self, goal):
        return goal in self.options

    def remove_from_options(self, goal):
        self.options.remove(goal)


class GNetManager:
    def __init__(
        self,
        env,
        gnet_goals,
        last_num_goals=75,
        max_exploration_rate=0.95,
        min_exploration_rate=0.1,
    ):
        self.gnet_goals = gnet_goals
        self.env = env
        self.gnet_state = ""
        self.goal_paths = []

        # These are used for exploration
        self.goal_successes = {}
        for start_goal in self.gnet_goals.keys():
            self.goal_successes[start_goal] = {}
            if "goal_selection_options" in self.gnet_goals[start_goal].keys():
                for selected_goal in self.gnet_goals[start_goal][
                    "goal_selection_options"
                ]:
                    self.goal_successes[start_goal][selected_goal] = np.array([])
        # Last number of goals to consider when calculating success rate
        self.last_num_goals = last_num_goals
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate

    def reset(self):
        self.goal_paths = []
        if self.gnet_goals["start"]["output_arc_type"] == "concurrent":
            for option in self.gnet_goals["start"]["goal_selection_options"]:
                self.add_path("start", [option])
        elif self.gnet_goals["start"]["output_arc_type"] == "choice":
            self.add_path("start", self.gnet_goals["start"]["goal_selection_options"])

    def set_state(self, gnet_goal, gnet_goal_achieved):
        if self.gnet_goals[gnet_goal_achieved]["output_arc_type"] == "choice":
            path = self.get_path(gnet_goal, gnet_goal_achieved)
            if path == None:
                self.goal_paths.append(
                    GoalNetPath(
                        gnet_goal, self.gnet_goals[gnet_goal]["goal_selection_options"]
                    )
                )
            else:
                path.current_gnet_goal = gnet_goal_achieved
                path.options = self.gnet_goals[gnet_goal_achieved][
                    "goal_selection_options"
                ]
                path.wait_list = []
                path.waiting = 0

        elif self.gnet_goals[gnet_goal_achieved]["output_arc_type"] == "concurrent":
            # Remove the previous path
            for path in self.goal_paths:
                if path.goal_in_options(gnet_goal_achieved):
                    self.goal_paths.pop(self.goal_paths.index(path))
                    break

            for next_gnet in self.gnet_goals[gnet_goal_achieved][
                "goal_selection_options"
            ]:
                self.goal_paths.append(GoalNetPath(gnet_goal_achieved, [next_gnet]))

        elif (
            self.gnet_goals[gnet_goal_achieved]["output_arc_type"] == "synchronization"
        ):
            sync_goal = self.gnet_goals[gnet_goal_achieved]["goal_selection_options"][0]
            found = False
            for path in self.goal_paths:
                if sync_goal in path.options:
                    found = True
                    path.waiting -= 1
                    path.wait_list.append(gnet_goal_achieved)
                    path.current_gnet_goal = gnet_goal_achieved
                    break

            # Add to path
            if not found:
                path = self.get_path(gnet_goal, gnet_goal_achieved)
                path.current_gnet_goal = gnet_goal_achieved
                path.options = self.gnet_goals[gnet_goal_achieved][
                    "goal_selection_options"
                ]
                path.wait_list.append(gnet_goal_achieved)
                path.waiting = self.gnet_goals[sync_goal]["input_arc_count"] - 1

            # Remove current path
            else:
                if self.get_path(gnet_goal, gnet_goal_achieved):
                    self.goal_paths.pop(
                        self.goal_paths.index(
                            self.get_path(gnet_goal, gnet_goal_achieved)
                        )
                    )

    # Generate goal mask and code based on current goal paths
    def generate_goal_mask(self):
        mask_code = [0 for i in range(len(self.gnet_goals))]
        goalnet_code = [0 for i in range(len(self.gnet_goals))]
        for path in self.goal_paths:
            goalnet_code[self.gnet_goals[path.current_gnet_goal]["code"].index(1)] = 1
            for goal in path.wait_list:
                goalnet_code[self.gnet_goals[goal]["code"].index(1)] = 1
            if path.waiting == 0:
                for option in path.options:
                    mask_code[self.gnet_goals[option]["code"].index(1)] = 1

        mask = []
        for x in mask_code:
            if x == 0:
                mask.append(-float("Inf"))
            else:
                mask.append(0)

        return goalnet_code, mask

    def get_parent_goal(self, target_goal):
        for path in self.goal_paths:
            if target_goal in path.options:
                return path.current_gnet_goal

    def generate_options_list(self):
        options = []
        for path in self.goal_paths:
            if path.waiting == 0:
                init = path.current_gnet_goal
                for option in path.options:
                    if option not in options:
                        options.append((init, option))

        return options

    def add_path(self, gnet_state, options, waiting_count=0):
        self.goal_paths.append(GoalNetPath(gnet_state, options, waiting_count))

    def remove_path(self, path):
        idx = self.goal_paths.index(path)
        self.goal_paths.pop(idx)

    # Get the path at a given gnet goal with a specified goal in its options
    def get_path(self, current_gnet_goal, gnet_goal_in_options):
        for path in self.goal_paths:
            if (
                path.goal_in_options(gnet_goal_in_options)
                and path.current_gnet_goal == current_gnet_goal
            ):
                return path

        return None

    def get_goal_selection_options(self):
        return self.gnet_goals[self.gnet_state]["goal_selection_options"]

    def update_success_rate(self, selected_goal, success, start_goal=None):
        if start_goal == None:
            start_goal = self.gnet_state
        self.goal_successes[start_goal][selected_goal] = np.append(
            self.goal_successes[start_goal][selected_goal][
                -(self.last_num_goals - 1) :
            ],
            [0 if not success else 1],
        )

    def get_exploration_rate(self, start_goal, selected_goal):
        if len(self.goal_successes[start_goal][selected_goal]) > 0:
            success_rate = self.goal_successes[start_goal][selected_goal].mean()
        else:
            success_rate = 0

        exp_rate = 1 - success_rate
        # Rescale the range of the exploration rate
        exp_rate = (self.max_exploration_rate - self.min_exploration_rate) * (
            exp_rate - 1
        ) + self.max_exploration_rate

        return exp_rate
