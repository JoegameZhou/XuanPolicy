from common import space2shape
from copy import deepcopy
from xuance_torch.policies import Policy_Inputs, Policy_Inputs_All
from xuance_torch.representations import Representation_Inputs, Representation_Inputs_All
from operator import itemgetter
import torch


def get_repre_in(args):
    representation_name = args.representation
    input_dict = deepcopy(Representation_Inputs_All)
    if isinstance(args.observation_space, dict):
        input_dict["input_shape"] = space2shape(args.observation_space[args.agent_keys[0]])
    else:
        input_dict["input_shape"] = space2shape(args.observation_space)

    if representation_name in ["Basic_MLP", "CoG_MLP"]:
        input_dict["hidden_sizes"] = args.representation_hidden_size
    else:
        if representation_name in ["Basic_CNN", "CoG_CNN"]:
            input_dict["kernels"] = args.kernels
            input_dict["strides"] = args.strides
            input_dict["filters"] = args.filters

    input_dict["normalize"] = None
    input_dict["initialize"] = torch.nn.init.orthogonal_
    input_dict["activation"] = torch.nn.Tanh
    input_dict["device"] = args.device

    input_list = itemgetter(*Representation_Inputs[representation_name])(input_dict)

    return list(input_list)


def get_policy_in(args, representation):
    policy_name = args.policy
    input_dict = deepcopy(Policy_Inputs_All)
    input_dict["action_space"] = args.action_space
    input_dict["representation"] = representation
    if policy_name in ["Basic_Q_network", "Duel_Q_network", "Noisy_Q_network", "C51_Q_network", "QR_Q_network", "CDQN_Policy", "LDQN_Policy", "CLDQN_Policy"]:
        input_dict["hidden_sizes"] = args.q_hidden_size
        if policy_name == "C51_Q_network":
            input_dict['vmin'] = args.vmin
            input_dict['vmax'] = args.vmax
            input_dict['atom_num'] = args.atom_num
        elif policy_name == "QR_Q_network":
            input_dict['quantile_num'] = args.quantile_num
    elif policy_name in ['PDQN_Policy', 'MPDQN_Policy', 'SPDQN_Policy']:
        input_dict['observation_space'] = args.observation_space
        input_dict['conactor_hidden_size'] = args.conactor_hidden_size
        input_dict['qnetwork_hidden_size'] = args.qnetwork_hidden_size
    else:
        input_dict["actor_hidden_size"] = args.actor_hidden_size
        if policy_name in ["Categorical_AC", "Categorical_PPG", "Gaussian_AC", "Gaussian_PPG", "DDPG_Policy", "TD3_Policy"]:
            input_dict["critic_hidden_size"] = args.critic_hidden_size
    input_dict["normalize"] = None
    input_dict["initialize"] = torch.nn.init.orthogonal_
    input_dict["activation"] = torch.nn.Tanh
    input_dict["device"] = args.device
    if policy_name == "Gaussian_Actor":
        input_dict["fixed_std"] = None
    input_list = itemgetter(*Policy_Inputs[policy_name])(input_dict)
    return list(input_list)


def get_policy_in_marl(args, representation, agent_keys, mixer=None, ff_mixer=None, qtran_mixer=None):
    policy_name = args.policy
    input_dict = deepcopy(Policy_Inputs_All)
    try: input_dict["state_dim"] = args.dim_state[0]
    except: input_dict["state_dim"] = None
    input_dict["action_space"] = args.action_space[agent_keys[0]]
    try: input_dict["n_agents"] = args.n_agents
    except: input_dict["n_agents"] = 1
    input_dict["representation"] = representation
    input_dict["mixer"] = mixer
    input_dict["ff_mixer"] = ff_mixer
    input_dict["qtran_mixer"] = qtran_mixer
    if policy_name in ["Basic_Q_network_marl", "Mixing_Q_network", "Weighted_Mixing_Q_network",
                       "Qtran_Mixing_Q_network", "MF_Q_network"]:
        input_dict["hidden_sizes"] = args.q_hidden_size
    else:
        input_dict["actor_hidden_size"] = args.actor_hidden_size
        try: input_dict["critic_hidden_size"] = args.critic_hidden_size
        except: input_dict["critic_hidden_size"] = None
    input_dict["normalize"] = None
    input_dict["initialize"] = None
    input_dict["activation"] = torch.nn.ReLU

    input_dict["device"] = args.device
    if policy_name == "Gaussian_Actor":
        input_dict["fixed_std"] = None
    input_list = itemgetter(*Policy_Inputs[policy_name])(input_dict)
    return list(input_list)
