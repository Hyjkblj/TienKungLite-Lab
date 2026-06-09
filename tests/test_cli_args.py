from __future__ import annotations

import argparse
import unittest

from real_lite_lab.cli_args import add_rsl_rl_args


class RslRlCliArgTests(unittest.TestCase):
    def test_policy_init_checkpoint_argument_is_available(self) -> None:
        parser = argparse.ArgumentParser()
        add_rsl_rl_args(parser)

        args = parser.parse_args(["--init_policy_checkpoint", "logs/stand_real_lite/run/model_2999.pt"])

        self.assertEqual(args.init_policy_checkpoint, "logs/stand_real_lite/run/model_2999.pt")


if __name__ == "__main__":
    unittest.main()
