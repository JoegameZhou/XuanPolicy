"""
MFQ: Mean Field Q-Learning
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
Creator: Wenzhang Liu (liu_wzh@foxmail.com)
"""
from xuance_torch.learners import *


class MFQ_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.temperature = config.temperature
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        super(MFQ_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)

    def get_boltzmann_policy(self, q):
        return self.softmax(q / self.temperature)

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        act_mean = torch.Tensor(sample['act_mean']).to(self.device)
        act_mean_next = torch.Tensor(sample['act_mean_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().view(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        act_mean = act_mean.unsqueeze(1).repeat([1, self.n_agents, 1])
        act_mean_next = act_mean_next.unsqueeze(1).repeat([1, self.n_agents, 1])
        _, _, q_eval = self.policy(obs, act_mean, IDs)
        q_eval_a = q_eval.gather(-1, actions.long().view([self.args.batch_size, self.n_agents, 1]))
        q_next = self.policy.target_Q(obs_next, act_mean_next, IDs)
        shape = q_next.shape
        pi = self.get_boltzmann_policy(q_next)
        v_mf = torch.bmm(q_next.view(-1, 1, shape[-1]), pi.unsqueeze(-1).view(-1, shape[-1], 1))
        v_mf = v_mf.view(*(list(shape[0:-1]) + [1]))
        if self.args.consider_terminal_states:
            q_target = rewards + (1 - terminals) * self.args.gamma * v_mf
        else:
            q_target = rewards + self.args.gamma * v_mf

        # calculate the loss function
        td_error = (q_eval_a - q_target.detach()) * agent_mask
        loss = (td_error ** 2).sum() / agent_mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("loss_Q", loss.item(), self.iterations)
        self.writer.add_scalar("predictQ", q_eval_a.mean().item(), self.iterations)
