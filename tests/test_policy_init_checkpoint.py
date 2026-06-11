from __future__ import annotations

import ast
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "train_real_lite.py"


def load_checkpoint_helpers():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    wanted_names = {"_compatible_policy_init_state"}
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "torch" for alias in node.names):
                selected_nodes.append(node)
            continue
        if getattr(node, "name", None) in wanted_names:
            selected_nodes.append(node)
    module = types.ModuleType("test_policy_init_checkpoint_module")
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(TRAIN_PATH), "exec"), module.__dict__)
    return module


class PolicyInitCheckpointTests(unittest.TestCase):
    def test_converts_scalar_std_checkpoint_to_log_std_policy(self) -> None:
        module = load_checkpoint_helpers()
        source_state = {
            "actor.0.weight": torch.ones(2, 2),
            "std": torch.full((3,), 0.5),
        }
        target_state = {
            "actor.0.weight": torch.zeros(2, 2),
            "log_std": torch.zeros(3),
        }

        compatible_state, skipped_keys, converted = module._compatible_policy_init_state(source_state, target_state)

        self.assertTrue(converted)
        self.assertEqual(skipped_keys, [])
        self.assertTrue(torch.equal(compatible_state["actor.0.weight"], source_state["actor.0.weight"]))
        self.assertTrue(torch.allclose(compatible_state["log_std"], torch.log(torch.full((3,), 0.5))))

    def test_skips_shape_mismatches(self) -> None:
        module = load_checkpoint_helpers()
        source_state = {"actor.0.weight": torch.ones(3, 2)}
        target_state = {"actor.0.weight": torch.zeros(2, 2)}

        compatible_state, skipped_keys, converted = module._compatible_policy_init_state(source_state, target_state)

        self.assertFalse(converted)
        self.assertEqual(compatible_state, {})
        self.assertEqual(skipped_keys, ["actor.0.weight"])


if __name__ == "__main__":
    unittest.main()
