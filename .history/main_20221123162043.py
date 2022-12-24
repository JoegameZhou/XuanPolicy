'''
Repo: XuanPolicy_v2.0

Example for single agent DRL:
$ python main.py --method dqn --env toy

Example for MARL with competitive sides:
$ python main.py --method vdn --env mpe
'''

import os
import sys
from common import get_command_config, get_arguments

algorithm_default = ["PerDQN"]
environment_default = "toy"

if __name__ == '__main__':
    args_command = sys.argv
    try:
        algorithm_name = get_command_config(args_command, "--method")
        print("Specified algorithm: ", algorithm_name)
    except IndexError:
        algorithm_name = algorithm_default
        print("Default algorithm: ", algorithm_name)
    try:
        env_name = get_command_config(args_command, "--env")
        print("Specified environment: ", env_name)
    except IndexError:
        env_name = environment_default
        print("Default environment: ", env_name)

    args_group, run_REGISTRY = get_arguments(os.path.dirname(__file__), "configs", algorithm_name, env_name)
    runner = run_REGISTRY[args_group[0].runner](args_group)
    runner.run()
