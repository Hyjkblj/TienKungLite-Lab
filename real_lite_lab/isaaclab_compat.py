from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import torch

try:
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg as _RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg as _RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg as _RslRlPpoAlgorithmCfg,
        export_policy_as_jit as _export_policy_as_jit,
        export_policy_as_onnx as _export_policy_as_onnx,
    )
except ImportError:
    _RslRlOnPolicyRunnerCfg = None
    _RslRlPpoActorCriticCfg = None
    _RslRlPpoAlgorithmCfg = None
    _export_policy_as_jit = None
    _export_policy_as_onnx = None

try:
    from isaaclab_tasks.utils import get_checkpoint_path as _get_checkpoint_path
except ImportError:
    _get_checkpoint_path = None

try:
    from isaaclab.utils.io import dump_yaml as _dump_yaml
except ImportError:
    _dump_yaml = None

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


def _config_to_dict(value: Any):
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _config_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_config_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [_config_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class _CfgBase:
    """Minimal fallback for Isaac Lab config objects used by this repository."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _field_names(cls) -> tuple[str, ...]:
        field_names: dict[str, None] = {}
        for base in reversed(cls.__mro__):
            if base is object:
                continue
            for name, value in vars(base).items():
                if name.startswith("_"):
                    continue
                if isinstance(value, (staticmethod, classmethod, property)):
                    continue
                if callable(value):
                    continue
                field_names[name] = None
        return tuple(field_names.keys())

    def to_dict(self) -> dict[str, Any]:
        return {name: _config_to_dict(getattr(self, name)) for name in self._field_names()}


if _RslRlOnPolicyRunnerCfg is not None:
    RslRlOnPolicyRunnerCfg = _RslRlOnPolicyRunnerCfg
    RslRlPpoActorCriticCfg = _RslRlPpoActorCriticCfg
    RslRlPpoAlgorithmCfg = _RslRlPpoAlgorithmCfg
else:

    class RslRlOnPolicyRunnerCfg(_CfgBase):
        pass


    class RslRlPpoActorCriticCfg(_CfgBase):
        class_name = "ActorCritic"
        init_noise_std = 1.0
        noise_std_type = "scalar"
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"


    class RslRlPpoAlgorithmCfg(_CfgBase):
        class_name = "PPO"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        normalize_advantage_per_mini_batch = False
        symmetry_cfg = None
        rnd_cfg = None


def _match_by_name(candidates: list[Path], pattern: str | None, kind: str) -> Path:
    if not candidates:
        raise FileNotFoundError(f"No {kind} candidates were found.")

    normalized_pattern = pattern or ".*"
    exact_matches = [candidate for candidate in candidates if candidate.name == normalized_pattern]
    if exact_matches:
        return sorted(exact_matches, key=lambda item: (item.stat().st_mtime, item.name))[-1]

    try:
        regex = re.compile(normalized_pattern)
    except re.error as exc:
        raise FileNotFoundError(f"Invalid {kind} pattern {normalized_pattern!r}: {exc}") from exc

    matches = [candidate for candidate in candidates if regex.fullmatch(candidate.name)]
    if not matches:
        raise FileNotFoundError(f"No {kind} matched pattern {normalized_pattern!r}.")
    return sorted(matches, key=lambda item: (item.stat().st_mtime, item.name))[-1]


def get_checkpoint_path(log_root_path: str | os.PathLike[str], load_run: str | None, load_checkpoint: str | None) -> str:
    if _get_checkpoint_path is not None:
        return _get_checkpoint_path(log_root_path, load_run, load_checkpoint)

    log_root = Path(log_root_path).expanduser().resolve()
    if not log_root.exists():
        raise FileNotFoundError(f"Log root directory does not exist: {log_root}")

    requested_run = Path(load_run).expanduser() if load_run else None
    if requested_run and requested_run.is_absolute() and requested_run.is_dir():
        run_dir = requested_run
    else:
        local_run = log_root / requested_run if requested_run else None
        if local_run and local_run.is_dir():
            run_dir = local_run
        else:
            run_candidates = [path for path in log_root.iterdir() if path.is_dir()]
            run_dir = _match_by_name(run_candidates, load_run, "run directory")

    requested_checkpoint = Path(load_checkpoint).expanduser() if load_checkpoint else None
    if requested_checkpoint and requested_checkpoint.is_absolute() and requested_checkpoint.is_file():
        checkpoint_path = requested_checkpoint
    else:
        local_checkpoint = run_dir / requested_checkpoint if requested_checkpoint else None
        if local_checkpoint and local_checkpoint.is_file():
            checkpoint_path = local_checkpoint
        else:
            checkpoint_candidates = [path for path in run_dir.rglob("*.pt") if path.is_file()]
            checkpoint_path = _match_by_name(checkpoint_candidates, load_checkpoint, "checkpoint file")

    return str(checkpoint_path)


def dump_yaml(filename: str | os.PathLike[str], cfg: Any):
    if _dump_yaml is not None:
        return _dump_yaml(filename, cfg)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _config_to_dict(cfg)

    with output_path.open("w", encoding="utf-8") as file:
        if _yaml is not None:
            _yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)


def _resolve_policy_network(policy: torch.nn.Module) -> torch.nn.Module:
    if hasattr(policy, "actor"):
        return policy.actor
    if hasattr(policy, "student"):
        return policy.student
    raise ValueError("Policy does not expose an actor/student network for export.")


def _infer_obs_dim(policy: torch.nn.Module) -> int:
    actor = _resolve_policy_network(policy)
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            return module.in_features
    raise ValueError("Unable to infer observation dimension from policy actor.")


class _TorchPolicyExporter(torch.nn.Module):
    """Local exporter that mirrors Isaac Lab's non-recurrent export path."""

    def __init__(self, policy: torch.nn.Module, normalizer: torch.nn.Module | None = None):
        super().__init__()
        if getattr(policy, "is_recurrent", False):
            raise NotImplementedError("Local export fallback currently supports only non-recurrent policies.")
        self.actor = copy.deepcopy(_resolve_policy_network(policy))
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(observations))


def export_policy_as_jit(
    policy: torch.nn.Module,
    normalizer: torch.nn.Module | None = None,
    path: str | os.PathLike[str] = ".",
    filename: str = "policy.pt",
):
    if _export_policy_as_jit is not None:
        return _export_policy_as_jit(policy, normalizer=normalizer, path=path, filename=filename)

    export_path = Path(path)
    export_path.mkdir(parents=True, exist_ok=True)
    exporter = _TorchPolicyExporter(policy, normalizer=normalizer).cpu().eval()
    scripted = torch.jit.script(exporter)
    scripted.save(str(export_path / filename))


def export_policy_as_onnx(
    policy: torch.nn.Module,
    normalizer: torch.nn.Module | None = None,
    path: str | os.PathLike[str] = ".",
    filename: str = "policy.onnx",
    opset_version: int = 17,
):
    if _export_policy_as_onnx is not None:
        return _export_policy_as_onnx(policy, normalizer=normalizer, path=path, filename=filename)

    export_path = Path(path)
    export_path.mkdir(parents=True, exist_ok=True)
    exporter = _TorchPolicyExporter(policy, normalizer=normalizer).cpu().eval()
    dummy_input = torch.zeros(1, _infer_obs_dim(policy), dtype=torch.float32)
    torch.onnx.export(
        exporter,
        dummy_input,
        str(export_path / filename),
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=opset_version,
    )
