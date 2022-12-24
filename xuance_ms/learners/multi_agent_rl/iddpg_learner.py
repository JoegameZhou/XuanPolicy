"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: MindSpore
Creator: Wenzhang Liu (liu_wzh@foxmail.com)
"""
from xuance_ms.learners import *


class IDDPG_Learner(LearnerMAS):
    class ActorNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(IDDPG_Learner.ActorNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._mean = ms.ops.ReduceMean(keep_dims=True)

        def construct(self, o, ids, agt_mask):
            _, actions_eval = self._backbone(o, ids)
            loss_a = -(self._backbone.critic(o, actions_eval, ids) * agt_mask).sum() / agt_mask.sum()
            return loss_a

    class CriticNetWithLossCell(nn.Cell):
        def __init__(self, backbone):
            super(IDDPG_Learner.CriticNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._loss = nn.MSELoss()

        def construct(self, o, a, ids, agt_mask, tar_q):
            q_eval = self._backbone.critic(o, a, ids)
            td_error = (q_eval - tar_q) * agt_mask
            loss_c = (td_error ** 2).sum() / agt_mask.sum()
            return loss_c

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: Sequence[nn.Optimizer],
                 scheduler: Sequence[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(IDDPG_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, modeldir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }

        # define mindspore trainers
        self.actor_loss_net = self.ActorNetWithLossCell(policy)
        self.actor_train = nn.TrainOneStepCell(self.actor_loss_net, self.optimizer['actor'])
        self.actor_train.set_train()
        self.critic_loss_net = self.CriticNetWithLossCell(policy)
        self.critic_train = nn.TrainOneStepCell(self.critic_loss_net, self.optimizer['critic'])
        self.critic_train.set_train()

    def update(self, sample):
        self.iterations += 1
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        obs_next = Tensor(sample['obs_next'])
        rewards = Tensor(sample['rewards'])
        terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1)
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))
        # calculate target values
        q_next = self.policy.target_critic(obs_next, self.policy.target_actor(obs_next, IDs), IDs)
        if self.args.consider_terminal_states:
            q_target = rewards + (1-terminals) * self.args.gamma * q_next
        else:
            q_target = rewards + self.args.gamma * q_next

        # calculate the loss and train
        loss_a = self.actor_train(obs, IDs, agent_mask)
        loss_c = self.critic_train(obs, actions, IDs, agent_mask, q_target)
        self.policy.soft_update(self.tau)

        lr_a = self.scheduler['actor'](self.iterations).asnumpy()
        lr_c = self.scheduler['critic'](self.iterations).asnumpy()
        self.writer.add_scalar("learning_rate_actor", lr_a, self.iterations)
        self.writer.add_scalar("learning_rate_critic", lr_c, self.iterations)
        self.writer.add_scalar("loss_actor", loss_a.asnumpy(), self.iterations)
        self.writer.add_scalar("loss_critic", loss_c.asnumpy(), self.iterations)
