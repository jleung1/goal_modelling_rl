from gym_miniworld.params import DEFAULT_PARAMS
from gym_miniworld.entity import Box
from gym_miniworld.miniworld import MiniWorldEnv
from gym import spaces
import numpy as np


class FourRoomsSubgoals3D(MiniWorldEnv):
    def __init__(
        self,
        max_episode_steps=300,
        preset=False,
        forward_step=0.6,
        turn_step=30,
        **kwargs
    ):
        self.preset = preset

        params = DEFAULT_PARAMS.no_random()
        params.set("forward_step", forward_step, forward_step - 0.1, forward_step + 0.1)
        params.set("turn_step", turn_step, turn_step - 10, turn_step + 10)

        super().__init__(max_episode_steps=max_episode_steps, params=params, **kwargs)

        self.action_space = spaces.Discrete(self.actions.move_back + 1)

    def _gen_world(self):
        # Taken from the gym miniworld fourrooms env
        # Top-left room
        room0 = self.add_rect_room(min_x=-7, max_x=-1, min_z=1, max_z=7)
        # Top-right room
        room1 = self.add_rect_room(min_x=1, max_x=7, min_z=1, max_z=7)
        # Bottom-right room
        room2 = self.add_rect_room(min_x=1, max_x=7, min_z=-7, max_z=-1)
        # Bottom-left room
        room3 = self.add_rect_room(min_x=-7, max_x=-1, min_z=-7, max_z=-1)

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        # Custom part
        self.reached_yellow_subgoal = False
        self.reached_blue_subgoal = False

        if self.preset:
            self.blue_subgoal = self.place_entity(Box(color="blue"), room=self.rooms[1])
            self.yellow_subgoal = self.place_entity(
                Box(color="yellow"), room=self.rooms[2]
            )
            self.goal = self.place_entity(Box(color="green"), room=self.rooms[3])
            self.place_agent(dir=0, room=self.rooms[0])
        else:
            goal_agent_rooms = np.random.choice([0, 1, 2, 3], 2, replace=False)
            subgoal_rooms = np.random.choice([0, 1, 2, 3], 2)
            self.goal = self.place_entity(
                Box(color="green"), room=self.rooms[goal_agent_rooms[0]]
            )
            self.place_agent(room=self.rooms[goal_agent_rooms[1]])

            self.blue_subgoal = self.place_entity(
                Box(color="blue"), room=self.rooms[subgoal_rooms[0]]
            )
            while self.near(self.blue_subgoal, self.goal, 2.5) or self.near(
                self.blue_subgoal, self.agent, 2.5
            ):
                self.entities.remove(self.blue_subgoal)
                self.blue_subgoal = self.place_entity(
                    Box(color="blue"), room=self.rooms[subgoal_rooms[0]]
                )

            self.yellow_subgoal = self.place_entity(
                Box(color="yellow"), room=self.rooms[subgoal_rooms[1]]
            )
            while (
                self.near(self.yellow_subgoal, self.goal, 2.5)
                or self.near(self.yellow_subgoal, self.agent, 2.5)
                or self.near(self.yellow_subgoal, self.blue_subgoal, 2.5)
            ):
                self.entities.remove(self.yellow_subgoal)
                self.yellow_subgoal = self.place_entity(
                    Box(color="yellow"), room=self.rooms[subgoal_rooms[1]]
                )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.yellow_subgoal) and not self.reached_yellow_subgoal:
            self.reached_yellow_subgoal = True
            self.entities.remove(self.yellow_subgoal)
        elif self.near(self.blue_subgoal) and not self.reached_blue_subgoal:
            self.reached_blue_subgoal = True
            self.entities.remove(self.blue_subgoal)
        elif self.near(self.goal):
            done = True
            reward = 1 - 0.9 * (self.step_count / self.max_episode_steps)

        if done and not (self.reached_blue_subgoal and self.reached_yellow_subgoal):
            reward = 0

        return obs, reward, done, info

    # Replace the near function
    def near(self, ent0, ent1=None, limit=1.5):
        if ent1 == None:
            ent1 = self.agent

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        return dist < limit
