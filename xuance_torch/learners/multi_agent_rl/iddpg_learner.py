"""
Independent Deep Deterministic Policy Gradient (IDDPG)
Implementation: Pytorch
Creator: Wenzhang Liu (liu_wzh@foxmail.com)
"""
from xuance_torch.learners import *


class IDDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(IDDPG_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().view(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        q_eval = self.policy.critic(obs, actions, IDs)
        q_next = self.policy.target_critic(obs_next, self.policy.target_actor(obs_next, IDs), IDs)
        if self.args.consider_terminal_states:
            q_target = rewards + (1-terminals) * self.args.gamma * q_next
        else:
            q_target = rewards + self.args.gamma * q_next

        # calculate the loss function
        _, actions_eval = self.policy(obs, IDs)
        loss_a = -(self.policy.critic(obs, actions_eval, IDs) * agent_mask).sum() / agent_mask.sum()
        self.optimizer['actor'].zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.clip_grad)
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        td_error = (q_eval - q_target.detach()) * agent_mask
        loss_c = (td_error ** 2).sum() / agent_mask.sum()
        self.optimizer['critic'].zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.clip_grad)
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()

        self.policy.soft_update(self.tau)

        lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("learning_rate_actor", lr_a, self.iterations)
        self.writer.add_scalar("learning_rate_critic", lr_c, self.iterations)
        self.writer.add_scalar("loss_actor", loss_a.item(), self.iterations)
        self.writer.add_scalar("loss_critic", loss_c.item(), self.iterations)
        self.writer.add_scalar("predictQ", q_eval.mean().item(), self.iterations)
