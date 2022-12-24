from xuance_torch.learners import *


class PG_Learner(Learner):
    def __init__(self,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 ent_coef: float = 0.005,
                 clip_grad: Optional[float] = None):
        super(PG_Learner, self).__init__(policy, optimizer, scheduler, summary_writer, device, modeldir)
        self.ent_coef = ent_coef
        self.clip_grad = clip_grad

    def update(self, obs_batch, act_batch, ret_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch, device=self.device)
        ret_batch = torch.as_tensor(ret_batch, device=self.device)
        outputs, a_dist = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)

        a_loss = -(ret_batch * log_prob).mean()
        e_loss = a_dist.entropy().mean()

        loss = a_loss - self.ent_coef * e_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("actor-loss", a_loss.item(), self.iterations)
        self.writer.add_scalar("entropy", e_loss.item(), self.iterations)
        self.writer.add_scalar("learning_rate", lr, self.iterations)
