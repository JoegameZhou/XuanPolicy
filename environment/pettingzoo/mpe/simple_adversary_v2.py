from envs.pettingzoo.utils.conversions import parallel_wrapper_fn

from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.simple_adversary import Scenario


class raw_env(SimpleEnv):
    def __init__(self, N=2, max_cycles=25, continuous_actions=True):
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata['name'] = "simple_adversary_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
