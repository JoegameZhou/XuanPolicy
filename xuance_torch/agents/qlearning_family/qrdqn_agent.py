from xuance_torch.agents import *
class QRDQN_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: VecEnv,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.render = config.render
        self.comm = MPI.COMM_WORLD
        self.nenvs = envs.num_envs

        self.gamma = config.gamma
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range

        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.egreedy = config.start_greedy

        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        writer = SummaryWriter(config.logdir)
        memory = DummyOffPolicyBuffer(self.observation_space,
                                      self.action_space,
                                      self.representation_info_shape,
                                      self.auxiliary_info_shape,
                                      self.nenvs,
                                      config.nsize,
                                      config.batchsize)
        learner = QRDQN_Learner(policy,
                                optimizer,
                                scheduler,
                                writer,
                                config.device,
                                config.modeldir,
                                config.gamma,
                                config.sync_frequency)

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(QRDQN_Agent, self).__init__(envs, policy, memory, learner, writer, device, config.logdir, config.modeldir)

    def _process_observation(self, observations):
        if self.use_obsnorm:
            if isinstance(self.observation_space, Dict):
                for key in self.observation_space.spaces.keys():
                    observations[key] = np.clip(
                        (observations[key] - self.obs_rms.mean[key]) / (self.obs_rms.std[key] + EPS),
                        -self.obsnorm_range, self.obsnorm_range)
            else:
                observations = np.clip((observations - self.obs_rms.mean) / (self.obs_rms.std + EPS),
                                       -self.obsnorm_range, self.obsnorm_range)
            return observations
        return observations

    def _process_reward(self, rewards):
        if self.use_rewnorm:
            std = np.clip(self.ret_rms.std, 0.1, 100)
            return np.clip(rewards / std, -self.rewnorm_range, self.rewnorm_range)
        return rewards

    def _action(self, obs, egreedy):
        states, argmax_action, _, _ = self.policy(obs)
        random_action = np.random.choice(self.action_space.n, self.nenvs)
        if np.random.rand() < egreedy:
            action = random_action
        else:
            action = argmax_action.detach().cpu().numpy()
        for key in states.keys():
            states[key] = states[key].detach().cpu().numpy()
        return states, action

    def train(self, train_steps=10000):
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)
        obs = self.envs.reset()
        for step in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts = self._action(obs, self.egreedy)
            next_obs, rewards, dones, infos = self.envs.step(acts)
            if self.render: self.envs.render()

            self.memory.store(obs, acts, self._process_reward(rewards), dones, self._process_observation(next_obs),
                              states, {})
            if step > self.start_training and step % self.train_frequency == 0:
                # training
                obs_batch, act_batch, rew_batch, terminal_batch, next_batch, _, _ = self.memory.sample()
                self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            self.egreedy = self.egreedy - (self.start_greedy - self.end_greedy) / train_steps

            for i in range(self.nenvs):
                if dones[i] == True:
                    self.ret_rms.update(returns[i:i + 1])
                    self.writer.add_scalars("returns-episode", {"env-%d" % i: scores[i]}, episodes[i])
                    self.writer.add_scalars("returns-step", {"env-%d" % i: scores[i]}, step)
                    scores[i] = 0
                    returns[i] = 0
                    episodes[i] += 1
            if step % 50000 == 0 or step == train_steps - 1:
                self.save_model()
                np.save(self.modeldir + "/obs_rms.npy",
                        {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

    def test(self, test_steps=10000, episode=0):
        self.load_model(self.modeldir)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)
        obs = self.envs.reset()
        for _ in tqdm(range(test_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts = self._action(obs, egreedy=0.0)
            next_obs, rewards, dones, infos = self.envs.step(acts)
            self.envs.render()

            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs

            for i in range(self.nenvs):
                if dones[i] == True:
                    scores[i], returns[i] = 0, 0

    def evaluate(self):
        pass
