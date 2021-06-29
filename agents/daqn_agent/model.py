import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base.model import *


class DaqnDQN(nn.Module):
    def __init__(self, num_goals, goal_space_size, action_space, env_name, device):
        super(DaqnDQN, self).__init__()
        self.device = device
        self.env_name = env_name
        self.frame_stack = 4

        if self.env_name == "two_keys":
            # 13x13 -> 12x12
            self.conv1 = nn.Conv2d(3, 32, 2)
            # 12x12 -> 11x11
            self.conv2 = nn.Conv2d(32, 64, 2)
            # 11x11 -> 10x10
            self.conv3 = nn.Conv2d(64, 64, 2)

            self.fc1 = nn.Linear(10 * 10 * 64 + num_goals + goal_space_size, 512)
            self.fc2 = nn.Linear(512, action_space)
        elif self.env_name == "four_rooms_3d":
            # 60x80 -> 30x40
            self.block1 = Block(3 * self.frame_stack, 32)
            # 30x40 -> 15x20
            self.block2 = Block(32, 64)
            # 15x20 -> 8x10
            self.block3 = Block(64, 64)

            self.fc1 = nn.Linear(
                8 * 10 * 64 + 2 + num_goals + 8 * self.frame_stack + (self.frame_stack - 1),
                512,
            )
            self.fc2 = nn.Linear(512, action_space)

        elif self.env_name == "ai2thor_kitchen":
            # 100x100 -> 50x50
            self.block1 = Block(4*self.frame_stack, 32)
            # 50x50 -> 25x25
            self.block2 = Block(32, 32)
            # 25x25 -> 13x13
            self.block3 = Block(32, 64)
            # 13x13 -> 7x7
            self.block4 = Block(64, 128)

            self.fc1 = nn.Linear(
                7 * 7 * 128 + 2 + num_goals+ 8 * self.frame_stack + (self.frame_stack - 1),
                512,
            )
            self.fc2 = nn.Linear(512, action_space)

        self.relu = nn.ReLU()

    def forward(self, x, goal, code, x_extra=None, actions=None):
        if self.env_name == "two_keys":
            batch_size = x.size(0)
            x = x.to(self.device)

            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))

            x = x.view(batch_size, -1)

            if x.dim() != code.dim():
                code = code.unsqueeze(0).to(self.device)
            if x.dim() != goal.dim():
                goal = goal.unsqueeze(0).to(self.device)

            x = torch.cat([x, goal.float(), code], dim=1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        if self.env_name == "four_rooms_3d" or self.env_name == "ai2thor_kitchen":
            batch_size = x.size(0)
            x = x.to(self.device)

            x = self.block1(x)
            x = self.block2(x)
            if self.env_name == "four_rooms_3d":
                x = self.relu(self.block3(x))
            else:
                x = self.block3(x)
                x = self.relu(self.block4(x))

            x = x.view(batch_size, -1)
            x = torch.cat([x, goal, code, x_extra, actions], dim=1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        return x
