import numpy as np

from xuance_torch.agents import *
import gym
from gym import spaces


class PDQN_Agent(Agent):
    def __init__(self,
                 config: Namespace,
                 envs: Toy_Env,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Optional[Sequence[torch.optim.lr_scheduler._LRScheduler]] = None,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.config = config
        self.envs = envs
        self.comm = MPI.COMM_WORLD
        self.render = config.render

        self.gamma = config.gamma
        self.use_obsnorm = config.use_obsnorm
        self.use_rewnorm = config.use_rewnorm
        self.obsnorm_range = config.obsnorm_range
        self.rewnorm_range = config.rewnorm_range

        self.train_frequency = config.training_frequency
        self.start_training = config.start_training
        self.start_noise = config.start_noise
        self.end_noise = config.end_noise
        self.noise_scale = config.start_noise

        self.observation_space = envs.observation_space.spaces[0]
        old_as = envs.action_space
        num_disact = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                        old_as.spaces[1].spaces[i].high, dtype=np.float32) for i in range(0, num_disact))))
        self.representation_info_shape = {'state': (envs.observation_space.spaces[0].shape)}
        self.auxiliary_info_shape = {}
        self.nenvs = 1
        self.epsilon = 0.1
        self.buffer_action_space = spaces.Box(np.zeros(4), np.ones(4), dtype=np.float64)

        writer = SummaryWriter(config.logdir)
        memory = DummyOffPolicyBuffer(self.observation_space,
                                      self.buffer_action_space,
                                      self.representation_info_shape,
                                      self.auxiliary_info_shape,
                                      self.nenvs,
                                      config.nsize,
                                      config.batchsize)
        learner = PDQN_Learner(policy,
                               optimizer,
                               scheduler,
                               writer,
                               config.device,
                               config.modeldir,
                               config.gamma,
                               config.tau)

        self.num_disact = self.action_space.spaces[0].n
        self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact+1)])
        self.conact_size = int(self.conact_sizes.sum())

        self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
        self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
        super(PDQN_Agent, self).__init__(envs, policy, memory, learner, writer, device, config.logdir, config.modeldir)

    def _process_observation(self, observations):
        if self.use_obsnorm:
            if isinstance(self.observation_space, gym.spaces.Dict):
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

    def _action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            con_actions = self.policy.con_action(obs)
            rnd = np.random.rand()
            if rnd < self.epsilon:
                disaction = np.random.choice(self.num_disact)
            else:
                q = self.policy.Qeval(obs.unsqueeze(0), con_actions.unsqueeze(0))
                q = q.detach().cpu().data.numpy()
                disaction = np.argmax(q)

        con_actions = con_actions.cpu().data.numpy()
        offset = np.array([self.conact_sizes[i] for i in range(disaction)], dtype=int).sum()
        conaction = con_actions[offset:offset+self.conact_sizes[disaction]]

        return disaction, conaction, con_actions

    def pad_action(self, disaction, conaction):
        con_actions = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
        con_actions[disaction][:] = conaction
        return (disaction, con_actions)

    def train(self, train_steps=10000):
        episodes = np.zeros((self.nenvs,), np.int32)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)
        obs, _ = self.envs.reset()
        for step in tqdm(range(train_steps)):
            disaction, conaction, con_actions = self._action(obs)
            action = self.pad_action(disaction, conaction)
            (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
            if self.render: self.envs.render()
            acts = np.concatenate(([disaction], con_actions), axis=0).ravel()
            state = {'state': obs}
            self.memory.store(obs, acts, rewards, terminal, next_obs, state, {})
            if step > self.start_training and step % self.train_frequency == 0:
                obs_batch, act_batch, rew_batch, terminal_batch, next_batch, _, _ = self.memory.sample()
                self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)
            scores += rewards
            returns = self.gamma * returns + rewards
            obs = next_obs
            self.noise_scale = self.start_noise - (self.start_noise - self.end_noise) / train_steps
            if step % 50000 == 0 or step == train_steps - 1:
                self.save_model()
                np.save(self.modeldir + "/obs_rms.npy",
                        {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

    def test(self, test_steps=10000, load_model=None):
        self.load_model(self.modeldir)
        scores = np.zeros((self.nenvs,), np.float32)
        returns = np.zeros((self.nenvs,), np.float32)
        obs = self.envs.reset()
        for _ in tqdm(range(test_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            states, acts, = self._action(obs, 0.0)
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
