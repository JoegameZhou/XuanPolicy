from abc import ABC, abstractmethod
from gym.spaces import Space, Box, Discrete, Dict
from argparse import Namespace
from mpi4py import MPI
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import *
from common import *
from xuance_torch.learners import *
from xuance_torch.policies import *
from xuance_torch.utils import *
from xuance_torch.policies import REGISTRY as REGISTRY_Policy
from xuance_torch.utils.input_reformat import get_repre_in, get_policy_in_marl
from xuance_torch.representations import REGISTRY as REGISTRY_Representation

from .agent import Agent
from .agents_marl import MARLAgents, RandomAgents, get_total_iters

'''
Single-Agent DRL algorithms
'''
from .policy_gradient.pg_agent import PG_Agent
from .policy_gradient.a2c_agent import A2C_Agent
from .policy_gradient.ppoclip_agent import PPOCLIP_Agent
from .policy_gradient.ppokl_agent import PPOKL_Agent
from .policy_gradient.ppg_agent import PPG_Agent
from .policy_gradient.ddpg_agent import DDPG_Agent
from .policy_gradient.td3_agent import TD3_Agent

from .qlearning_family.dqn_agent import DQN_Agent
from .qlearning_family.dueldqn_agent import DuelDQN_Agent
from .qlearning_family.ddqn_agent import DDQN_Agent
from .qlearning_family.noisydqn_agent import NoisyDQN_Agent

'''
Multi-Agent DRL algorithms
'''
from .multi_agent_rl.iql_agents import IQL_Agents
from .multi_agent_rl.vdn_agents import VDN_Agents
from .multi_agent_rl.qmix_agents import QMIX_Agents
from .multi_agent_rl.wqmix_agents import WQMIX_Agents
from .multi_agent_rl.qtran_agents import QTRAN_Agents
from .multi_agent_rl.dcg_agents import DCG_Agents
from .multi_agent_rl.vdac_agents import VDAC_Agents
from .multi_agent_rl.coma_agents import COMA_Agents
from .multi_agent_rl.iddpg_agents import IDDPG_Agents
from .multi_agent_rl.maddpg_agents import MADDPG_Agents
from .multi_agent_rl.mfq_agents import MFQ_Agents
from .multi_agent_rl.mfac_agents import MFAC_Agents
from .multi_agent_rl.mappoclip_agents import MAPPO_Clip_Agents
from .multi_agent_rl.mappokl_agents import MAPPO_KL_Agents
from .multi_agent_rl.isac_agents import ISAC_Agents
from .multi_agent_rl.masac_agents import MASAC_Agents

REGISTRY = {
    "PG": PG_Agent,
    "A2C": A2C_Agent,
    "PPO_Clip": PPOCLIP_Agent,
    "PPO_KL": PPOKL_Agent,
    "PPG": PPG_Agent,
    "DDPG": DDPG_Agent,
    "TD3": TD3_Agent,
    "DQN": DQN_Agent,
    "Duel_DQN": DuelDQN_Agent,
    "DDQN": DDQN_Agent,
    "NoisyDQN": NoisyDQN_Agent,

    "RANDOM": RandomAgents,
    "IQL": IQL_Agents,
    "VDN": VDN_Agents,
    "QMIX": QMIX_Agents,
    "CWQMIX": WQMIX_Agents,
    "OWQMIX": WQMIX_Agents,
    "QTRAN_base": QTRAN_Agents,
    "QTRAN_alt": QTRAN_Agents,
    "DCG": DCG_Agents,
    "DCG_S": DCG_Agents,
    "VDAC": VDAC_Agents,
    "COMA": COMA_Agents,
    "IDDPG": IDDPG_Agents,
    "MADDPG": MADDPG_Agents,
    "MFQ": MFQ_Agents,
    "MFAC": MFAC_Agents,
    "MAPPO_Clip": MAPPO_Clip_Agents,
    "MAPPO_KL": MAPPO_KL_Agents,
    "ISAC": ISAC_Agents,
    "MASAC": MASAC_Agents
}
