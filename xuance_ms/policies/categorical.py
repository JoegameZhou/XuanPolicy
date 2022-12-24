from xuance_ms.policies import *
from xuance_ms.utils import *
from mindspore.nn.probability.distribution import Categorical


class ActorNet(nn.Cell):
    class Sample(nn.Cell):
        def __init__(self):
            super(ActorNet.Sample, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, probs: ms.tensor):
            return self._dist.sample(probs=probs).astype("int32")

    class LogProb(nn.Cell):
        def __init__(self):
            super(ActorNet.LogProb, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, value, probs):
            return self._dist.log_prob(value=value, probs=probs)

    class Entropy(nn.Cell):
        def __init__(self):
            super(ActorNet.Entropy, self).__init__()
            self._dist = Categorical(dtype=ms.float32)

        def construct(self, probs):
            return self._dist.entropy(probs=probs)

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, nn.Softmax, None)[0])
        self.model = nn.SequentialCell(*layers)
        self.sample = self.Sample()
        self.log_prob = self.LogProb()
        self.entropy = self.Entropy()

    def construct(self, x: ms.Tensor):
        return self.model(x)


class CriticNet(nn.Cell):
    def __init__(self,
                 state_dim: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None)[0])
        self.model = nn.SequentialCell(*layers)

    def construct(self, x: ms.Tensor):
        return self.model(x)[:, 0]


class ActorCriticPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Discrete)
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        v = self.critic(outputs['state'])
        return outputs, a, v


class ActorPolicy(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Discrete)
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)

    def construct(self, observation: ms.tensor):
        outputs = self.representation(observation)
        a = self.actor(outputs['state'])
        return outputs, a


class PPGActorCritic(nn.Cell):
    def __init__(self,
                 action_space: Space,
                 representation: ModuleType,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., ms.Tensor]] = None,
                 activation: Optional[ModuleType] = None
                 ):
        assert isinstance(action_space, Discrete)
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.policy_representation = representation
        self.critic_representation = copy.deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation)

    def construct(self, observation: ms.tensor):
        policy_outputs = self.policy_representation(observation) 
        critic_outputs = self.critic_representation(observation)
        a = self.actor(policy_outputs['state'])
        v = self.critic(critic_outputs['state'])
        aux_v = self.aux_critic(policy_outputs['state'])
        return policy_outputs, a, v, aux_v

