import ai2thor.controller
import copy
import numpy as np
from skimage.transform import resize

int_to_action = [
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "CloseObject",
    "ToggleObjectOff",
]


class KitchenEnv:
    def __init__(
        self, location=None, visibility=1.0, quality="Very Low", split=26, max_steps=500
    ):
        if location:
            self.controller = ai2thor.controller.Controller(
                local_executable_path=location,
                quality=quality,
                visibilityDistance=visibility,
                renderDepthImage=True,
            )
        else:
            self.controller = ai2thor.controller.Controller(
                quality=quality, visibilityDistance=visibility, renderDepthImage=True
            )

        self.train_floors = [i for i in range(1, 31)]
        self.train_floor_epoch = []
        self.floor_prefix = "FloorPlan"

        self.total_steps = 0
        self.max_steps = max_steps
        self.last_meta = None
        self.last_frame = None
        self.last_depth = None
        self.max_depth = 5.0

        self.fridge_pos = []
        self.light_pos = []

    def process_obs(self, obs, depth):
        state = resize(obs, (100, 100, 3))
        # Normalize
        depth[depth > self.max_depth] = self.max_depth
        depth = resize(depth, (100, 100))
        depth = np.array(depth * (255.0 / self.max_depth), dtype=np.uint8)
        depth = np.expand_dims(depth, -1)

        state = np.concatenate((state, depth), axis=-1)

        return state

    def reset(self, train=True, idx=-1, seed=-1):
        self.total_steps = 0
        if train:
            if not self.train_floor_epoch:
                self.train_floor_epoch = copy.deepcopy(self.train_floors)
                np.random.shuffle(self.train_floor_epoch)

            floor_id = self.train_floor_epoch.pop()
            floor_id = str(floor_id)
            scene = self.floor_prefix + floor_id
        else:
            scene = self.floor_prefix + str(idx)

        self.controller.reset(scene=scene)
        self.controller.step(
            action="SetObjectStates",
            SetObjectStates={
                "objectType": "Fridge",
                "stateChange": "openable",
                "isOpen": True,
            },
            renderImage=False,
        )
        if seed == -1:
            seed = np.random.randint(10000)
        self.controller.step(
            action="InitialRandomSpawn", randomSeed=seed, renderImage=False
        )

        event = self.controller.step("GetReachablePositions")
        positions = event.metadata["reachablePositions"]
        pos = np.random.choice(positions)

        if train:
            y = np.random.choice([0.0, 90.0, 180.0, 270.0])
            event = self.controller.step(
                "TeleportFull",
                x=pos["x"],
                y=y,
                z=pos["z"],
                rotation=dict(x=0.0, y=y, z=0.0),
            )

        self.last_meta = event.metadata
        self.last_frame = event.frame
        self.last_depth = event.depth_frame

        fridge = None
        light = None
        for i in self.last_meta["objects"]:
            if i["name"].find("Fridge") == 0:
                fridge = i
            elif i["name"].find("LightSwitch") == 0:
                light = i

        self.fridge_pos = [
            fridge["position"]["x"],
            fridge["position"]["y"],
            fridge["position"]["z"],
        ]
        self.light_pos = [
            light["position"]["x"],
            light["position"]["y"],
            light["position"]["z"],
        ]

        obs = event.frame
        depth = event.depth_frame
        state = self.process_obs(obs, depth)
        return state

    # Take a step in the environment and calculate the rewards
    def step(self, action_int):
        action = int_to_action[action_int]
        if action == "CloseObject" or action == "ToggleObjectOff":
            # Try to perform action on the closest object
            closest_obj_id = None
            closest_obj_dist = -1
            for obj in self.last_meta["objects"]:
                if (
                    (
                        (action == "CloseObject" and obj["openable"] and obj["isOpen"])
                        or (
                            action == "ToggleObjectOff"
                            and obj["toggleable"]
                            and obj["isToggled"]
                        )
                    )
                    and obj["visible"]
                    and (obj["distance"] < closest_obj_dist or closest_obj_dist == -1)
                ):
                    closest_obj_id = obj["objectId"]
                    closest_obj_dist = obj["distance"]
            if closest_obj_dist > -1:
                event = self.controller.step(action, objectId=closest_obj_id)
            else:
                # Use the previous rendered image
                event = self.controller.step(action, renderImage=False)
                event.frame = self.last_frame
                event.depth_frame = self.last_depth
        else:
            event = self.controller.step(action)
        self.total_steps += 1
        self.last_meta = event.metadata
        self.last_frame = event.frame
        self.last_depth = event.depth_frame

        fridge = None
        light = None
        for i in event.metadata["objects"]:
            if i["name"].find("Fridge") == 0:
                fridge = i
            elif i["name"].find("LightSwitch") == 0:
                light = i

        obs = event.frame
        depth = event.depth_frame
        state = self.process_obs(obs, depth)
        goal = [int(fridge["isOpen"]), int(light["isToggled"])]
        if goal[-1] == 0:
            done = True
            if goal[0] == 0:
                reward = 1 - 0.9 * (self.total_steps / self.max_steps)
            else:
                reward = 0
        else:
            done = False
            reward = 0

        if self.total_steps >= self.max_steps:
            done = True

        return state, reward, done, event.metadata
