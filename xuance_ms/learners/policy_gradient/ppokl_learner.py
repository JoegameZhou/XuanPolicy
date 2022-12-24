from xuance_torch.learners import *
class PPOKL_Learner(Learner):
    def __init__(self,
                 policy:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 scheduler:Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer:Optional[SummaryWriter] = None,
                 device: Optional[Union[int,str,torch.device]] = None,
                 modeldir: str = "./",
                 vf_coef: float = 0.25,
                 ent_coef: float = 0.005,
                 target_kl: float = 0.1):
        super(PPOKL_Learner,self).__init__(policy,optimizer,scheduler,summary_writer,device,modeldir)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.kl_coff = 1.0
    
    def update(self,obs_batch,act_batch,ret_batch,adv_batch,old_dist_batch):
        self.iterations += 1
        act_batch = torch.as_tensor(act_batch,device=self.device)
        ret_batch = torch.as_tensor(ret_batch,device=self.device)
        adv_batch = torch.as_tensor(adv_batch,device=self.device)
        
        
        outputs,a_dist,v_pred = self.policy(obs_batch)
        
        log_prob = a_dist.log_prob(act_batch)
        old_log_prob = old_dist.log_prob(act_batch)
        ratio = (log_prob - old_log_prob).exp().float()
        a_loss = - (ratio * adv_batch).mean()
        kl_loss = self.kl_coff*a_dist.kl_divergence(old_dist).mean()
        c_loss = F.mse_loss(v_pred,ret_batch)
        e_loss = a_dist.entropy().mean()
        loss = a_loss + self.kl_coff * kl_loss - self.ent_coef*e_loss + self.vf_coef*c_loss
        
        if kl_loss < self.target_kl * 0.5:
            self.kl_coff = self.kl_coff / 1.5
        elif kl_loss > self.target_kl * 2.0:
            self.kl_coff = self.kl_coff * 1.5
        self.kl_coff = np.clip(self.kl_coff,0,1,10)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        cr = ((ratio<1-self.clip_range).sum()+(ratio>1+self.clip_range).sum())/ratio.shape[0]
        self.writer.add_scalar("actor-loss",a_loss.item(),self.iterations)
        self.writer.add_scalar("critic-loss",c_loss.item(),self.iterations)
        self.writer.add_scalar("kl-loss",kl_loss.item(),self.iterations)
        self.writer.add_scalar("kl-coff",self.kl_coff,self.iterations)
        self.writer.add_scalar("entropy",e_loss.item(),self.iterations)
        self.writer.add_scalar("learning_rate",lr,self.iterations)
        self.writer.add_scalar("predict_value",v_pred.mean().item(),self.iterations)
        
         
        
             
        
         