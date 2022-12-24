import os
import numpy as np
import scipy.signal
import yaml
import itertools
from gym.spaces import Space, Dict
from argparse import Namespace
from typing import Sequence
from types import SimpleNamespace as SN
from copy import deepcopy

EPS = 1e-8


def recursive_dict_update(basic_dict, target_dict):
    out_dict = deepcopy(basic_dict)
    for key, value in target_dict.items():
        if isinstance(value, dict):
            out_dict[key] = recursive_dict_update(out_dict.get(key, {}), value)
        else:
            out_dict[key] = value
    return out_dict


def get_config(dir_name, args_name):
    with open(os.path.join(dir_name, args_name + ".yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, args_name + ".yaml error: {}".format(exc)
    return config_dict


def get_command_config(params, arg_name):
    values = None
    if arg_name == "--method":
        values = []
        for _i, _v in enumerate(params):
            if arg_name in _v:
                values.append(params[_i + 1])
    else:
        for _i, _v in enumerate(params):
            if arg_name in _v:
                values = params[_i + 1]
    if values:
        return values
    else:
        raise IndexError("No method is contained in command!")


def get_arguments(main_path, folder_name, agent_name, env_name):
    config_path = os.path.join(main_path, folder_name)
    # Get the defaults from basic.yaml
    config_basic = get_config(config_path, "basic")
    config_basic['env_name'] = env_name
    config_algorithm = [get_config(os.path.join(config_path, agent), env_name) for agent in agent_name]

    config = [recursive_dict_update(config_basic, config_i) for config_i in config_algorithm]
    args = [SN(**config_i) for config_i in config]

    if config_basic['dl_toolbox'] == "torch":
        from xuance_torch.runners import REGISTRY as run_REGISTRY
        notation = "_th/"
    elif config_basic['dl_toolbox'] == "mindspore":
        from xuance_ms.runners import REGISTRY as run_REGISTRY
        from mindspore import context
        notation = "_ms/"
        if args[0].device != "Auto":
            if args[0].device == "cpu": args[0].device = "CPU"
            context.set_context(device_target=args[0].device)
        # context.set_context(enable_graph_kernel=True)
        context.set_context(mode=context.GRAPH_MODE)  # 静态图（断点无法进入）
        # context.set_context(mode=context.PYNATIVE_MODE)  # 动态图（便于调试）
    elif config_basic['dl_toolbox'] == "tensorlayer":
        from xuance_tl.runners import REGISTRY as run_REGISTRY
        notation = "_tl/"
    else:
        if config_basic['dl_toolbox'] == '':
            raise AttributeError("You have to assign a deep learning toolbox")
        else:
            raise AttributeError("Cannot find a deep learning toolbox named " + args[i_alg].dl_toolbox)

    for i_alg in range(len(agent_name)):
        args[i_alg].agent_name = agent_name[i_alg]
        args[i_alg].modeldir = os.path.join(main_path, args[i_alg].modeldir + notation + args[i_alg].env_id + '/')
        args[i_alg].logdir = args[i_alg].logdir + notation + args[i_alg].env_id + '/'

    if args[0].test_mode:
        args[0].parallels = 1
    return args, run_REGISTRY

## Above is created by Wenzhang Liu. ###


def create_directory(path):
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1, len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def space2shape(observation_space: Space):
    if isinstance(observation_space, Dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    else:
        return observation_space.shape


def dict_reshape(keys, dict_list: Sequence[dict]):
    results = {}
    for key in keys():
        results[key] = np.array([element[key] for element in dict_list], np.float32)
    return results


def discount_cumsum(x, discount=0.99):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def merge_iterators(self, *iters):
    itertools.chain(*iters)
