from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from rsl_rl.env import VecEnv

if TYPE_CHECKING:
    from .walk_cfg import RealLiteWalkAgentCfg, RealLiteWalkEnvCfg


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(
        self,
        name: str,
        task_class: VecEnv,
        env_cfg: "RealLiteWalkEnvCfg",
        train_cfg: "RealLiteWalkAgentCfg",
    ):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name: str):
        return copy.deepcopy(self.env_cfgs[name]), copy.deepcopy(self.train_cfgs[name])


task_registry = TaskRegistry()
