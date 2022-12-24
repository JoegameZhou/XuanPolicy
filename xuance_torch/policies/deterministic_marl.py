import copy

import torch

from xuance_torch.policies import *
from xuance_torch.utils import *
from xuance_torch.representations import Basic_Identical
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
from itertools import chain


class BasicQhead(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQhead, self).__init__()
        layers_ = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.model = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class BasicQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(BasicQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class MFQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(MFQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)

    def forward(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def sample_actions(self, logits: torch.Tensor):
        dist = Categorical(logits=logits)
        return dist.sample()

    def target_Q(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        return self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)


class MixingQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.eval_Qtot = mixer
        self.target_Qtot = copy.deepcopy(self.eval_Qtot)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_Qhead(q_inputs)

    def Q_tot(self, q, states=None):
        return self.eval_Qtot(q, states)

    def target_Q_tot(self, q, states=None):
        return self.target_Qtot(q, states)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)


class Weighted_MixingQnetwork(MixingQnetwork):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 ff_mixer: Optional[QMIX_FF_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(Weighted_MixingQnetwork, self).__init__(action_space, n_agents, representation, mixer, hidden_size,
                                                      normalize, initialize, activation, device)
        self.eval_Qhead_centralized = copy.deepcopy(self.eval_Qhead)
        self.target_Qhead_centralized = copy.deepcopy(self.eval_Qhead_centralized)
        self.q_feedforward = ff_mixer
        self.target_q_feedforward = copy.deepcopy(self.q_feedforward)

    def q_centralized(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.eval_Qhead_centralized(q_inputs)

    def target_q_centralized(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_Qhead_centralized(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead_centralized.parameters(), self.target_Qhead_centralized.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.q_feedforward.parameters(), self.target_q_feedforward.parameters()):
            tp.data.copy_(ep)


class Qtran_MixingQnetwork(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 qtran_mixer: Optional[QTRAN_base] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(Qtran_MixingQnetwork, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     hidden_size, normalize, initialize, activation, device)
        self.target_Qhead = copy.deepcopy(self.eval_Qhead)
        self.qtran_net = qtran_mixer
        self.target_qtran_net = copy.deepcopy(qtran_mixer)
        self.q_tot = mixer

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def target_Q(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        return outputs, self.target_Qhead(q_inputs)

    def copy_target(self):
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.qtran_net.parameters(), self.target_qtran_net.parameters()):
            tp.data.copy_(ep)


class DCG_policy(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 global_state_dim: int,
                 representation: Optional[Basic_Identical],
                 utility: Optional[DCG_utility] = None,
                 payoffs: Optional[DCG_payoff] = None,
                 dcgraph: Optional[Coordination_Graph] = None,
                 hidden_size_bias: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(DCG_policy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.utility = utility
        self.target_utility = copy.deepcopy(self.utility)
        self.payoffs = payoffs
        self.target_payoffs = copy.deepcopy(self.payoffs)
        self.graph = dcgraph
        self.dcg_s = False
        if hidden_size_bias is not None:
            self.dcg_s = True
            self.bias = BasicQhead(global_state_dim, 1, 0, hidden_size_bias,
                                   normalize, initialize, activation, device)
            self.target_bias = copy.deepcopy(self.bias)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
        evalQ = self.eval_Qhead(q_inputs)
        argmax_action = evalQ.argmax(dim=-1, keepdim=False)
        return outputs, argmax_action, evalQ

    def copy_target(self):
        for ep, tp in zip(self.utility.parameters(), self.target_utility.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.payoffs.parameters(), self.target_payoffs.parameters()):
            tp.data.copy_(ep)
        if self.dcg_s:
            for ep, tp in zip(self.bias.parameters(), self.target_bias.parameters()):
                tp.data.copy_(ep)


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Tanh, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class CriticNet(nn.Module):
    def __init__(self,
                 independent: bool,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        if independent:
            input_shape = (state_dim + action_dim + n_agents,)
        else:
            input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)


class Basic_DDPG_policy(nn.Module):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        assert isinstance(action_space, Box)
        super(Basic_DDPG_policy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes

        self.actor_net = ActorNet(representation.output_shapes['state'][0], n_agents, self.action_dim,
                                  actor_hidden_size, normalize, initialize, activation, device)
        self.critic_net = CriticNet(True, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_actor = list(self.representation.parameters()) + list(self.actor_net.parameters())
        self.parameters_critic = self.critic_net.parameters()

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        act = self.actor_net(actor_in)
        return outputs, act

    def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)

    def target_actor(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        actor_in = torch.concat([outputs['state'], agent_ids], dim=-1)
        return self.target_actor_net(actor_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MADDPG_policy(Basic_DDPG_policy):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        assert isinstance(action_space, Box)
        super(MADDPG_policy, self).__init__(action_space, n_agents, representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initialize, activation, device)
        self.critic_net = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_critic = self.critic_net.parameters()

    def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)


class Attention_CriticNet(nn.Module):
    def __init__(self,
                 independent: bool,
                 state_dim: int,
                 n_agents: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(Attention_CriticNet, self).__init__()
        layers = []
        if independent:
            input_shape = (state_dim + action_dim + n_agents,)
        else:
            input_shape = (state_dim * n_agents + action_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, initialize, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.model(x)

class AttentionCritic(nn.Module):
    def __init__(self,
                 independent: bool,
                 state_dim: int,
                 n_agent: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 norm_in=True,
                 attend_heads=1):
        super(AttentionCritic, self).__init__()
        assert (hidden_sizes[0] % attend_heads) == 0
        # if independent:
        #     input_shape = (state_dim + action_dim + n_agent, )
        # else:
        #     input_shape = (state_dim * n_agent + action_dim * n_agent + n_agent)
        self.attend_heads = attend_heads
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agent = n_agent
        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()
        # iterate over agents
        idim = state_dim * n_agent + action_dim * n_agent + n_agent
        odim = action_dim
        encoder = nn.Sequential()
        if norm_in:
            encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                        affine=False))
        encoder.add_module('enc_fc1', nn.Linear(idim, hidden_sizes[0]))
        encoder.add_module('enc_nl', nn.LeakyReLU())
        self.critic_encoders.append(encoder)
        critic = nn.Sequential()
        critic.add_module('critic_fc1', nn.Linear(2 * hidden_sizes[0],
                                                  hidden_sizes[0]))
        critic.add_module('critic_nl', nn.LeakyReLU())
        critic.add_module('critic_fc2', nn.Linear(hidden_sizes[0], odim))
        self.critics.append(critic)

        state_encoder = nn.Sequential()
        if norm_in:
            state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                        state_dim * n_agent, affine=False))
        state_encoder.add_module('s_enc_fc1', nn.Linear(state_dim * n_agent,
                                                        hidden_sizes[0]))
        state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
        self.state_encoders.append(state_encoder)

        attend_dim = hidden_sizes[0] // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_sizes[0], attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_sizes[0], attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_sizes[0],
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = inps[:, :, :self.state_dim * self.n_agent]
        actions = inps[:, :, self.state_dim * self.n_agent: (self.state_dim + self.action_dim) * self.n_agent]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # s_encodings = [self.state_encoders[a_i](states[:, a_i, :]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(self.n_agent)]
        all_attend_logits = [[] for _ in range(self.n_agent)]
        all_attend_probs = [[] for _ in range(self.n_agent)]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(self.n_agent), range(self.n_agent), curr_head_selectors[0]):
                keys = [k for j, k in enumerate(curr_head_keys[0]) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values[0]) if j != a_i]
                # calculate attention across agents
                # attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),  torch.stack(keys).permute(1, 2, 0))
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1), keys[0].view(keys[0].shape[0], 1, -1))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[0])
                attend_weights = F.softmax(scaled_attend_logits, dim=0)
                other_values = (values[0].view(values[0].shape[0], 1, -1) * attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits[:, :, 0])
                all_attend_probs[i].append(attend_weights[:, :, 0])
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(range(1)):
            head_entropies = [(-((probs + 1e-8).log() * probs).sum(1) .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[0][i], *other_all_values[i][0]), dim=0)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[:, a_i, a_i * self.action_dim:a_i * self.action_dim + self.action_dim].max(dim=0, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

class MAAC_policy(Basic_DDPG_policy):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        assert isinstance(action_space, Box)
        super(MAAC_policy, self).__init__(action_space, n_agents, representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initialize, activation, device)
        self.critic_net = AttentionCritic(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size,  norm_in=True, attend_heads=1)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_critic = self.critic_net.parameters()

    def critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)

class MATD3_policy(Basic_DDPG_policy):
    def __init__(self,
                 action_space: Space,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        assert isinstance(action_space, Box)
        super(MATD3_policy, self).__init__(action_space, n_agents, representation,
                                            actor_hidden_size, critic_hidden_size,
                                            normalize, initialize, activation, device)
        self.critic_net_A = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.critic_net_B = CriticNet(False, representation.output_shapes['state'][0], n_agents, self.action_dim,
                                      critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic_net_A = copy.deepcopy(self.critic_net_A)
        self.target_critic_net_B = copy.deepcopy(self.critic_net_B)
        # self.parameters_critic = self.critic_net.parameters()

    def Qpolicy(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, (qa + qb) / 2.0

    def Qtarget(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        min_q = torch.minimum(qa, qb)
        return outputs_n, min_q

    def Qaction(self, observation: torch.Tensor, actions: torch.Tensor, agent_ids: torch.Tensor):
        bs = observation.shape[0]
        outputs_n = self.representation(observation)['state'].view(bs, 1, -1).expand(-1, self.n_agents, -1)
        actions_n = actions.view(bs, 1, -1).expand(-1, self.n_agents, -1)
        critic_in = torch.concat([outputs_n, actions_n, agent_ids], dim=-1)
        qa = self.critic_net_A(critic_in)
        qb = self.critic_net_B(critic_in)
        return outputs_n, torch.cat((qa, qb), dim=-1)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net_A.parameters(), self.target_critic_net_A.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net_B.parameters(), self.target_critic_net_B.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)