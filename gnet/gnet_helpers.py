# Goal Net helper functions
def generate_gnet_codes(gnet_goals):
    num_goals = len(gnet_goals.keys())
    code = [0 for x in range(num_goals)]
    code[0] = 1
    for i, key in enumerate(gnet_goals.keys()):
        gnet_goals[key]["code"] = code
        code = [0] + code[:-1]

    return gnet_goals


def generate_gnet_masks(gnet_goals):
    num_goals = len(gnet_goals.keys())
    for key in gnet_goals.keys():
        mask = [-float("Inf") for x in range(num_goals)]
        if key == "goal":
            continue
        for goal in gnet_goals[key]["goal_selection_options"]:
            idx = gnet_goals[goal]["code"].index(1)
            mask[idx] = 0
        gnet_goals[key]["mask"] = mask

    return gnet_goals


def generate_idx2goalnet(gnet_goals):
    num_goals = len(gnet_goals.keys())
    idx2goalnet = ["" for x in range(num_goals)]
    for key in gnet_goals.keys():
        idx = gnet_goals[key]["code"].index(1)
        idx2goalnet[idx] = key

    return idx2goalnet


def prepare_gnet_goals(gnet_goals):
    gnet_goals = generate_gnet_codes(gnet_goals)
    gnet_goals = generate_gnet_masks(gnet_goals)

    idx2goalnet = generate_idx2goalnet(gnet_goals)

    return gnet_goals, idx2goalnet
