import numpy as np
import torch
import gym
import argparse
import json
import pickle

from env.two_keys import *
from gnet.gnet_manager_two_keys import TwoKeysGNetManager
from gnet.gnet_manager_four_rooms_subgoals_3d import FourRoomSubgoals3DGNetManager
from gnet.gnet_manager_kitchen import KitchenGNetManager
from gnet.gnet_helpers import prepare_gnet_goals
from gym_minigrid.wrappers import *

from agents.gnet_agent.gnet_agent import GNetAgent
from agents.gnet_agent.replay_memory import GNetActionReplayMemory, GNetGoalReplayMemory
from agents.gnet_agent.model import GNetActionDQN, GNetGoalDQN
from agents.daqn_agent.model import DaqnDQN
from agents.daqn_agent.replay_memory import DaqnReplayMemory
from agents.daqn_agent.daqn_agent import DaqnAgent
from agents.qrm_agent.replay_memory import QrmReplayMemory
from agents.qrm_agent.model import QrmDQN
from agents.qrm_agent.qrm_agent import QrmAgent

parser = argparse.ArgumentParser(
    description="Goal-Oriented Deep Reinforcement Learning"
)

parser.add_argument("--episodes", default=50000)
parser.add_argument("--seed", default=0)
parser.add_argument("--batch_size", default=32)
parser.add_argument(
    "--env",
    choices=["two_keys", "four_rooms_3d", "ai2thor_kitchen"],
    default="two_keys",
)
parser.add_argument(
    "--agent", choices=["gnet", "gnet_without_ga", "daqn", "qrm"], default="gnet"
)

parser.add_argument("--gpu", default=0)
parser.add_argument("--start_frame", default=0)
parser.add_argument("--episode", default=0)
# Loading
parser.add_argument(
    "--load", dest="load", action="store_true", help="Load existing checkpoint"
)
parser.add_argument(
    "--xvfb", dest="xvfb", action="store_true", help="Use xvfb (for AI2-THOR)"
)
parser.add_argument("--save", dest="save", action="store_true")
parser.add_argument("--preset", dest="preset", action="store_true")

args = parser.parse_args()

random_seed = int(args.seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

if args.xvfb:
    import pyvirtualdisplay

    _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
    _ = _display.start()

from env.four_rooms_subgoals_3d import *
from env.ai2thor_kitchen_env import *

gpu = "cuda:" + str(args.gpu)
device = torch.device(gpu)

episodes = int(args.episodes)

if args.env == "two_keys":
    env = TwoKeysEnv(preset=args.preset, seed=random_seed)
    env = FullyObsWrapper(env)

    f = open("./gnet/gnet_two_keys.json")
    gnet_goals = json.load(f)
    f.close()

    gnet_goals, idx2goalnet = prepare_gnet_goals(gnet_goals)
    gnet_manager = TwoKeysGNetManager(env, gnet_goals)

    replay_size = 1000000

elif args.env == "four_rooms_3d":
    env = FourRoomsSubgoals3D(preset=args.preset)
    if args.agent != "qrm":
        filename = "./gnet/gnet_four_rooms_subgoals_3d.json"
    else:
        filename = "./gnet/gnet_four_rooms_subgoals_3d_qrm.json"

    f = open(filename)
    gnet_goals = json.load(f)
    f.close()

    gnet_goals, idx2goalnet = prepare_gnet_goals(gnet_goals)
    gnet_manager = FourRoomSubgoals3DGNetManager(env, gnet_goals)

    replay_size = 250000

elif args.env == "ai2thor_kitchen":
    env = KitchenEnv(max_steps=200)
    f = open("./gnet/gnet_ai2thor_kitchen.json")
    gnet_goals = json.load(f)
    f.close()

    gnet_goals, idx2goalnet = prepare_gnet_goals(gnet_goals)
    gnet_manager = KitchenGNetManager(env, gnet_goals)

    replay_size = 200000

action_space_size = gnet_manager.action_space_size
goal_space_size = gnet_manager.goal_space_size

if args.agent == "gnet" or args.agent == "gnet_without_ga":
    memory = GNetActionReplayMemory(
        replay_size,
        goal_space_size,
        device,
        args.batch_size,
        args.env,
        (args.agent == "gnet"),
        load=args.load,
    )
    goal_memory = GNetGoalReplayMemory(
        10000, len(idx2goalnet), device, args.batch_size, args.env, load=args.load
    )

    model = GNetActionDQN(
        action_space_size, goal_space_size, device, args.env, (args.agent == "gnet")
    ).to(device)
    goal_model = GNetGoalDQN(len(idx2goalnet), device, args.env).to(device)
    agent = GNetAgent(
        env,
        gnet_manager,
        model,
        goal_model,
        memory,
        goal_memory,
        action_space_size,
        goal_space_size,
        len(idx2goalnet),
        idx2goalnet,
        (args.agent == "gnet"),
        args.env,
        device,
    )

elif args.agent == "daqn":
    memory = DaqnReplayMemory(
        replay_size, len(idx2goalnet), device, args.batch_size, args.env, load=args.load
    )
    # For the AMDP
    state_space_size = 2 ** (goal_space_size - 1)
    goal_q_values = np.zeros((state_space_size, action_space_size + 1))
    model = DaqnDQN(len(idx2goalnet), 2, action_space_size, args.env, device).to(device)

    agent = DaqnAgent(
        env,
        gnet_manager,
        model,
        goal_q_values,
        memory,
        action_space_size,
        goal_space_size,
        len(idx2goalnet),
        idx2goalnet,
        args.env,
        device,
    )

elif args.agent == "qrm":
    memory = QrmReplayMemory(
        replay_size, len(idx2goalnet), device, args.batch_size, args.env, load=args.load
    )
    model = QrmDQN(action_space_size, len(idx2goalnet), args.env, device).to(device)
    agent = QrmAgent(
        env,
        gnet_manager,
        model,
        memory,
        action_space_size,
        goal_space_size,
        len(idx2goalnet),
        idx2goalnet,
        args.env,
        device,
    )

if args.load:
    agent.action_model.load_state_dict(
        torch.load("./save/models_and_data/low_level.pt")
    )
    agent.clone_action_model.load_state_dict(
        torch.load("./save/models_and_data/low_level_clone.pt")
    )
    agent.action_opt.load_state_dict(
        torch.load("./save/models_and_data/low_level_opt.pt")
    )

    file = open("./save/models_and_data/goal_successes.pkl", "rb")
    agent.gnet_manager.goal_successes = pickle.load(file)
    file.close()

    if args.agent == "gnet" or args.agent == "gnet_without_ga":
        agent.goal_model.load_state_dict(
            torch.load("./save/models_and_data/high_level.pt")
        )
        agent.clone_goal_model.load_state_dict(
            torch.load("./save/models_and_data/high_level_clone.pt")
        )
        agent.goal_opt.load_state_dict(
            torch.load("./save/models_and_data/high_level_opt.pt")
        )
    elif args.agent == "daqn":
        infile = open("./save/models_and_data/goal_q_table.npy", "rb")
        goal_q_values = np.load(infile)
        infile.close()

        agent.goal_q_table = goal_q_values

result = agent.run(
    episodes,
    train=True,
    episode=int(args.episode),
    start_frame=int(args.start_frame),
    save_checkpoints=args.save,
)

if args.save:
    agent.save(result)
