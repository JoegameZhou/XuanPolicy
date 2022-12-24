from .runner_basic import *
from xuance_ms.agents import get_total_iters
from xuance_ms.representations import REGISTRY as REGISTRY_Representation
from xuance_ms.agents import REGISTRY as REGISTRY_Agent
from xuance_ms.policies import REGISTRY as REGISTRY_Policy
from xuance_ms.utils.input_reformat import get_repre_in, get_policy_in
import itertools
from mindspore.nn import Adam
from mindspore.nn.learning_rate_schedule import ExponentialDecayLR as lr_decay_model


class Runner_DRL(Runner_Base):
    def __init__(self, args):
        self.args = args[0]
        self.agent_name = self.args.agent
        super(Runner_DRL, self).__init__(self.args)

        self.args.observation_space = self.envs.observation_space
        self.args.action_space = self.envs.action_space

        input_representation = get_repre_in(self.args)
        representation = REGISTRY_Representation[self.args.representation](*input_representation)

        input_policy = get_policy_in(self.args, representation)
        policy = REGISTRY_Policy[self.args.policy](*input_policy)

        if self.agent_name in ["DDPG", "TD3"]:
            actor_lr_scheduler = lr_decay_model(learning_rate=self.args.actor_learning_rate,
                                                decay_rate=0.5,
                                                decay_steps=get_total_iters(self.agent_name, self.args))
            critic_lr_scheduler = lr_decay_model(learning_rate=self.args.critic_learning_rate,
                                                 decay_rate=0.5,
                                                 decay_steps=get_total_iters(self.agent_name, self.args))
            actor_optimizer = Adam(policy.actor_params, actor_lr_scheduler, eps=1e-5)
            if self.agent_name == "TD3":
                critic_optimizer = Adam(itertools.chain(policy.criticA.trainable_params(),
                                                        policy.criticB.trainable_params()),
                                        critic_lr_scheduler, eps=1e-5)
            else:
                critic_optimizer = Adam(policy.critic.trainable_params(), critic_lr_scheduler, eps=1e-5)
            self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy,
                                                         {'actor': actor_optimizer, 'critic': critic_optimizer},
                                                         {'actor': actor_lr_scheduler, 'critic': critic_lr_scheduler})
        else:
            lr_scheduler = lr_decay_model(learning_rate=self.args.learning_rate,
                                          decay_rate=0.5,
                                          decay_steps=get_total_iters(self.agent_name, self.args)
                                          )
            optimizer = Adam(policy.trainable_params(), lr_scheduler, eps=1e-5)
            self.agent = REGISTRY_Agent[self.agent_name](self.args, self.envs, policy, optimizer, lr_scheduler)

    def run(self):
        self.agent.test() if self.args.test_mode else self.agent.train(self.args.training_steps)
