from gym_minigrid.minigrid import *
from gym_minigrid.roomgrid import *
from gym_minigrid.register import register
import numpy as np


class TwoKeysEnv(RoomGrid):
    def __init__(self, room_size=7, seed=None, preset=True):
        self.preset = preset
        self.object_list = ["yellow_key", "green_key", "yellow_door", "green_door"]

        super().__init__(
            room_size=room_size,
            num_rows=2,
            num_cols=2,
            max_steps=300,
            seed=seed,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.goal_room = [1, 1]

        self.goal = Goal()

        # Add doors
        room = self.room_grid[1][1]
        room.door_pos[2] = (room.door_pos[2][0], room.top[1] + (room.size[1] // 2))
        room.door_pos[3] = (room.top[0] + (room.size[0] // 2), room.door_pos[3][1])

        if self.preset:
            yellow = 3
            green = 2
            self.yellow_key = Key("yellow")
            self.green_key = Key("green")
            self.goal_pos = self.place_obj(self.goal, (7, 10), (1, 1))
            self.yellow_key_pos = self.place_obj(self.yellow_key, (3, 7), (1, 1))
            self.green_key_pos = self.place_obj(self.green_key, (5, 4), (1, 1))
            super(RoomGrid, self).place_agent((1, 4), (1, 1), 0)
        else:
            yellow = self._rand_int(0, 2)
            green = (1 - yellow) + 2
            yellow += 2
            _, self.goal_pos = self.place_in_room(
                self.goal_room[0], self.goal_room[1], self.goal
            )
            self.yellow_key, self.yellow_key_pos = self.add_obj_not_goal_room(
                "key", "yellow"
            )
            self.green_key, self.green_key_pos = self.add_obj_not_goal_room(
                "key", "green"
            )
            self.add_obj_not_goal_room("agent")

        self.yellow_door, self.yellow_door_pos = self.add_door(
            self.goal_room[0],
            self.goal_room[1],
            door_idx=yellow,
            color="yellow",
            locked=True,
        )
        self.green_door, self.green_door_pos = self.add_door(
            self.goal_room[0],
            self.goal_room[1],
            door_idx=green,
            color="green",
            locked=True,
        )

        # Remove walls between non-goal rooms
        for j in range(0, self.num_rows):
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                if room.pos_inside(self.goal_pos[0], self.goal_pos[1]):
                    continue
                for k, neighbor in enumerate(room.neighbors):
                    if neighbor and not neighbor.pos_inside(
                        self.goal_pos[0], self.goal_pos[1]
                    ):
                        if not room.doors[k]:
                            self.remove_wall(i, j, k)

    # Add an object in a random room other than the goal room
    def add_obj_not_goal_room(self, kind, color=None):
        room = [self._rand_int(0, 2), self._rand_int(0, 2)]
        while room == self.goal_room:
            room = [self._rand_int(0, 2), self._rand_int(0, 2)]

        if kind == "agent":
            self.place_agent(room[0], room[1])
            pos = [room[0], room[1]]
            obj = None
        else:
            obj, pos = self.add_object(room[0], room[1], kind, color)
        return obj, pos

    def step(self, action):
        # We don't use action 4 in this env (drop item)
        if action == 4:
            action = 5
        obs, reward, done, info = MiniGridEnv.step(self, action)

        return obs, reward, done, info
