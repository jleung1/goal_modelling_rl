import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base.model import *


class GNetActionDQN(nn.Module):
    def __init__(self, action_space, goal_space, device, env_name, use_ga):
        super(GNetActionDQN, self).__init__()
        self.device = device
        self.env_name = env_name
        self.use_ga = use_ga
        self.frame_stack = 4

        if self.env_name == "two_keys":
            # 13x13 -> 12x12
            self.conv1 = nn.Conv2d(3, 32, 2)
            # 12x12 -> 11x11
            self.conv2 = nn.Conv2d(32, 64, 2)
            # 10x10 -> 10x10
            self.conv3 = nn.Conv2d(64, 64, 2)

            if self.use_ga:
                self.fc1 = nn.Linear(goal_space * 2, 32)
            else:
                self.fc1 = nn.Linear(goal_space, 32)
            self.fc2 = nn.Linear(10 * 10 * 64 + 32, 512)
            self.fc3 = nn.Linear(512, action_space)

        elif self.env_name == "four_rooms_3d":
            # 60x80 -> 30x40
            self.block1 = Block(3 * self.frame_stack, 32)
            # 30x40 -> 15x20
            self.block2 = Block(32, 64)
            # 15x20 -> 8x10
            self.block3 = Block(64, 64)

            if self.use_ga:
                self.fc1 = nn.Linear(
                    8 * 10 * 64 + goal_space + (goal_space + 6) * self.frame_stack + (self.frame_stack - 1),
                    512,
                )
            else:
                self.fc1 = nn.Linear(
                    8 * 10 * 64 + goal_space + 8 * self.frame_stack + (self.frame_stack - 1),
                    512,
                )
            self.fc2 = nn.Linear(512, action_space)

        elif self.env_name == "ai2thor_kitchen":
            # 100x100 -> 50x50
            self.block1 = Block(4 * self.frame_stack, 32)
            # 50x50 -> 25x25
            self.block2 = Block(32, 32)
            # 25x25 -> 13x13
            self.block3 = Block(32, 64)
            # 13x13 -> 7x7
            self.block4 = Block(64, 128)

            if self.use_ga:
                self.fc1 = nn.Linear(
                    7 * 7 * 128 + goal_space + (goal_space + 6) * self.frame_stack + (self.frame_stack - 1),
                    512,
                )
            else:
                self.fc1 = nn.Linear(
                    7 * 7 * 128 + goal_space + 8 * self.frame_stack + (self.frame_stack - 1),
                    512,
                )
            self.fc2 = nn.Linear(512, action_space)

        self.relu = nn.ReLU()

    def forward(self, x, goal, c_goal, actions=None):
        batch_size = x.size(0)

        if self.env_name == "two_keys":
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))

            x = x.view(batch_size, -1)

            if x.dim() != goal.dim():
                goal = goal.unsqueeze(0).to(self.device)

            if self.use_ga:
                if x.dim() != c_goal.dim():
                    c_goal = c_goal.unsqueeze(0).to(self.device)
                g = torch.cat([goal.float(), c_goal.float()], dim=1)
                g = self.relu(self.fc1(g))
            else:
                g = self.relu(self.fc1(goal.float()))

            x = torch.cat([x, g], dim=1)
            x = self.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            x = self.block1(x)
            x = self.block2(x)
            if self.env_name == "four_rooms_3d":
                x = self.relu(self.block3(x))
            else:
                x = self.block3(x)
                x = self.relu(self.block4(x))
            x = x.view(batch_size, -1)

            x = torch.cat([x, goal, c_goal, actions], dim=1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)

        return x


class GNetGoalDQN(nn.Module):
    def __init__(self, num_goals, device, env_name):
        super(GNetGoalDQN, self).__init__()
        self.device = device
        self.env_name = env_name

        if self.env_name == "two_keys":
            # 13x13 -> 12x12
            self.conv1 = nn.Conv2d(3, 32, 2)
            # 12x12 -> 11x11
            self.conv2 = nn.Conv2d(32, 64, 2)
            # 10x10 -> 10x10
            self.conv3 = nn.Conv2d(64, 64, 2)

            self.fc1 = nn.Linear(10 * 10 * 64 + num_goals, 256)
            self.fc2 = nn.Linear(256, num_goals)
        elif self.env_name == "four_rooms_3d":
            # 60x80 -> 27x37
            self.conv1 = nn.Conv2d(3, 16, 7, stride=2)
            # 27x37 -> 11x16
            self.conv2 = nn.Conv2d(16, 32, 6, stride=2)
            # 11x11 -> 9x14
            self.conv3 = nn.Conv2d(32, 32, 3)

            self.fc1 = nn.Linear(9 * 14 * 32 + num_goals + 8, 512)
            self.fc2 = nn.Linear(512, num_goals)

        self.relu = nn.ReLU()

    def forward(self, x, goalnet_code, mask, x_extra=None):
        batch_size = x.size(0)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(batch_size, -1)

        if x.dim() != goalnet_code.dim():
            goalnet_code = goalnet_code.unsqueeze(0).to(self.device)

        if self.env_name == "two_keys":
            x = torch.cat([x, goalnet_code.float()], dim=1)
        else:
            x_extra = x_extra.unsqueeze(0).to(self.device)
            x = torch.cat([x, goalnet_code.float(), x_extra], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x + mask.to(self.device)

        return x
