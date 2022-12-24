"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: Pytorch
Creator: Wenzhang Liu (liu_wzh@foxmail.com)
"""
from xuance_torch.learners import *


class QTRAN_Learner(LearnerMAS):
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
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(QTRAN_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        actions_onehot = self.onehot_action(actions, self.dim_act)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().view(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        hidden_n, _, q_eval = self.policy(obs, IDs)
        # get mask input
        actions_mask = agent_mask.repeat(1, 1, self.dim_act)
        hidden_mask = agent_mask.repeat(1, 1, hidden_n['state'].shape[-1])
        q_joint, v_joint = self.policy.qtran_net(hidden_n['state'] * hidden_mask,
                                                 actions_onehot * actions_mask)
        hidden_n_next, q_next_eval = self.policy.target_Q(obs_next.view([self.args.batch_size, self.n_agents, -1]), IDs)
        if self.args.double_q:
            _, actions_next_greedy, _ = self.policy(obs_next, IDs)
        else:
            actions_next_greedy = q_next_eval.argmax(dim=-1, keepdim=False)
        q_joint_next, _ = self.policy.target_qtran_net(hidden_n_next['state'] * hidden_mask,
                                                       self.onehot_action(actions_next_greedy,
                                                                          self.dim_act) * actions_mask)
        if self.args.consider_terminal_states:
            y_dqn = rewards + (1 - terminals) * self.args.gamma * q_joint_next
        else:
            y_dqn = rewards + self.args.gamma * q_joint_next
        loss_td = self.mse_loss(q_joint, y_dqn.detach())

        action_greedy = q_eval.argmax(dim=-1, keepdim=False)  # \bar{u}
        q_eval_greedy_a = q_eval.gather(-1, action_greedy.long().view([self.args.batch_size, self.n_agents, 1]))
        q_tot_greedy = self.policy.q_tot(q_eval_greedy_a * agent_mask)
        q_joint_greedy_hat, _ = self.policy.qtran_net(hidden_n['state'] * hidden_mask,
                                                      self.onehot_action(action_greedy, self.dim_act) * actions_mask)
        error_opt = q_tot_greedy - q_joint_greedy_hat.detach() + v_joint
        loss_opt = torch.mean(error_opt ** 2)

        q_eval_a = q_eval.gather(-1, actions.long().view([self.args.batch_size, self.n_agents, 1]))
        if self.args.agent == "QTRAN_base":
            q_tot = self.policy.q_tot(q_eval_a * agent_mask)
            q_joint_hat, _ = self.policy.qtran_net(hidden_n['state'] * hidden_mask,
                                                   actions_onehot * actions_mask)
            error_nopt = q_tot - q_joint_hat.detach() + v_joint
            error_nopt = error_nopt.clamp(max=0)
            loss_nopt = torch.mean(error_nopt ** 2)
        elif self.args.agent == "QTRAN_alt":
            q_tot_counterfactual = self.policy.qtran_net.counterfactual_values(q_eval, q_eval_a) * actions_mask
            q_joint_hat_counterfactual = self.policy.qtran_net.counterfactual_values_hat(hidden_n['state'] * hidden_mask,
                                                                                         actions_onehot * actions_mask)
            error_nopt = q_tot_counterfactual - q_joint_hat_counterfactual.detach() + v_joint.unsqueeze(dim=-1).repeat(
                1, self.n_agents, self.dim_act)
            error_nopt_min = torch.min(error_nopt, dim=-1).values
            loss_nopt = torch.mean(error_nopt_min ** 2)
        else:
            raise ValueError("Mixer {} not recognised.".format(self.args.agent))

        # calculate the loss function
        loss = loss_td + self.args.lambda_opt * loss_opt + self.args.lambda_nopt * loss_nopt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("loss_td", loss_td.item(), self.iterations)
        self.writer.add_scalar("loss_opt", loss_opt.item(), self.iterations)
        self.writer.add_scalar("loss_nopt", loss_nopt.item(), self.iterations)
        self.writer.add_scalar("loss", loss.item(), self.iterations)
        self.writer.add_scalar("predictQ", q_eval_a.mean().item(), self.iterations)
