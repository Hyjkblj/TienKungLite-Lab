from __future__ import annotations

from .env import RealLiteEnv
from .run_cfg import RealLiteRunAgentCfg, RealLiteRunEnvCfg
from .stand_cfg import RealLiteStandAgentCfg, RealLiteStandEnvCfg
from .task_registry import task_registry
from .upper_body_cfg import RealLiteUpperBodyAgentCfg, RealLiteUpperBodyEnvCfg
from .walk_cfg import RealLiteWalkAgentCfg, RealLiteWalkEnvCfg

_REGISTERED = False


def register_tasks():
    global _REGISTERED
    if _REGISTERED:
        return
    task_registry.register("walk_real_lite", RealLiteEnv, RealLiteWalkEnvCfg(), RealLiteWalkAgentCfg())
    task_registry.register("stand_real_lite", RealLiteEnv, RealLiteStandEnvCfg(), RealLiteStandAgentCfg())
    task_registry.register("run_real_lite", RealLiteEnv, RealLiteRunEnvCfg(), RealLiteRunAgentCfg())
    task_registry.register("upper_body_real_lite", RealLiteEnv, RealLiteUpperBodyEnvCfg(), RealLiteUpperBodyAgentCfg())
    _REGISTERED = True
