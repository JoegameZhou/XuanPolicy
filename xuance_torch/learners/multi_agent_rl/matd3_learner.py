"""
Multi-Agent TD3

"""
from xuance_torch.learners import *


class MATD3_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100,
                 delay: int = 3
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.delay = delay
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(MATD3_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic_A': optimizer[1],
            'critic_B': optimizer[2]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic_A': scheduler[1],
            'critic_B': scheduler[2]
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

        # train critic
        _, action_q = self.policy.Qaction(obs, actions, IDs)
        actions_next = self.policy.target_actor(obs, IDs)
        _, target_q = self.policy.Qtarget(obs, actions_next, IDs)
        if self.args.consider_terminal_states:
            q_target = rewards + (1 - terminals) * self.args.gamma * target_q
        else:
            q_target = rewards + self.args.gamma * target_q
        td_error = (action_q - q_target.detach()) * agent_mask
        loss_c = (td_error ** 2).sum() / agent_mask.sum()
        self.optimizer['critic_B'].zero_grad()
        self.optimizer['critic_A'].zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.clip_grad)
        self.optimizer['critic_A'].step()
        self.optimizer['critic_B'].step()
        if self.scheduler['critic_A'] is not None:
            self.scheduler['critic_A'].step()
            self.scheduler['critic_B'].step()

        # actor update
        if self.iterations % self.delay == 0:
            _, actions_eval = self.policy(obs, IDs)
            _, policy_q = self.policy.Qpolicy(obs, actions_eval, IDs)
            p_loss = -policy_q.mean()
            self.optimizer['actor'].zero_grad()
            p_loss.backward()
            self.optimizer['actor'].step()
            if self.scheduler is not None:
                self.scheduler['actor'].step()
            self.policy.soft_update(self.tau)

        lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        lr_c_A = self.optimizer['critic_A'].state_dict()['param_groups'][0]['lr']
        lr_c_B = self.optimizer['critic_B'].state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("learning_rate_actor", lr_a, self.iterations)
        self.writer.add_scalar("learning_rate_critic_A", lr_c_A, self.iterations)
        self.writer.add_scalar("learning_rate_critic_B", lr_c_B, self.iterations)
        if self.iterations % self.delay == 0:
            self.writer.add_scalar("loss_actor", p_loss.item(), self.iterations)
        self.writer.add_scalar("loss_critic_A", loss_c.item(), self.iterations)
        self.writer.add_scalar("loss_critic_B", loss_c.item(), self.iterations)
