"""
Value Decomposition Actor-Critic (VDAC)
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17353
Implementation: MindSpore
Creator: Wenzhang Liu (liu_wzh@foxmail.com)
"""
from xuance_ms.learners import *


class VDAC_Learner(LearnerMAS):
    class PolicyNetWithLossCell(nn.Cell):
        def __init__(self, backbone, vf_coef, ent_coef):
            super(VDAC_Learner.PolicyNetWithLossCell, self).__init__()
            self._backbone = backbone
            self._vf_coef = vf_coef
            self._ent_coef = ent_coef
            self.loss_c = nn.MSELoss()

        def construct(self, o, s, a, adv, ret, ids, agt_mask):
            _, act_probs, v_pred = self._backbone(o, ids)
            v_pred_tot = self._backbone.value_tot(v_pred * agt_mask, s)
            log_prob = self._backbone.actor.log_prob(value=a, probs=act_probs).reshape(adv.shape)
            entropy = self._backbone.actor.entropy(probs=act_probs).reshape(agt_mask.shape) * agt_mask

            loss_a = -(adv * log_prob * agt_mask).mean()
            loss_c = self.loss_c(logits=v_pred_tot, labels=ret)
            loss_e = entropy.mean()

            loss = loss_a + self._vf_coef * loss_c - self._ent_coef * loss_e
            return loss

    def __init__(self,
                 config: Namespace,
                 policy: nn.Cell,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.exponential_decay_lr] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        super(VDAC_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, modeldir)
        self.loss_net = self.PolicyNetWithLossCell(policy, config.vf_coef, config.ent_coef)
        self.policy_train = TrainOneStepCellWithGradClip(self.loss_net, optimizer,
                                                         clip_type=config.clip_type, clip_value=config.clip_grad)
        self.policy_train.set_train()

    def update(self, sample):
        self.iterations += 1
        state = Tensor(sample['state'])
        obs = Tensor(sample['obs'])
        actions = Tensor(sample['actions'])
        returns = Tensor(sample['values']).mean(axis=1)
        advantages = Tensor(sample['advantages'])
        agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
        batch_size = obs.shape[0]
        IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                               (batch_size, -1, -1))

        loss = self.policy_train(obs, state, actions, advantages, returns, IDs, agent_mask)

        # Logger
        lr = self.scheduler(self.iterations).asnumpy()
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("loss", loss.asnumpy(), self.iterations)
