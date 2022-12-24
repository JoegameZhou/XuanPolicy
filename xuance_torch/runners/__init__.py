from .runner_drl import Runner_DRL as DRL_runner
from .runner_marl import Runner as MARL_runner

REGISTRY = {
    "DRL": DRL_runner,
    "MARL": MARL_runner
}
