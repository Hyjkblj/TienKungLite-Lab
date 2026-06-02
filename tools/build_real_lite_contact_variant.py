from __future__ import annotations

import argparse
from pathlib import Path

from real_lite_lab.mjcf_contact_variants import build_toe_rail_contact_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Real Lite MJCF contact-geometry variant.")
    parser.add_argument(
        "--model",
        default="mjcf/real_lite.xml",
        help="Input MuJoCo XML model path.",
    )
    parser.add_argument(
        "--variant",
        choices=("toe_rails",),
        default="toe_rails",
        help="Contact variant to generate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output XML path. Defaults to '<input>.toe_rails.xml'.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

    if args.variant != "toe_rails":
        raise ValueError(f"Unsupported contact variant: {args.variant}")

    result = build_toe_rail_contact_model(model_path, output_path=output_path)
    print(f"[INFO] Wrote contact variant: {result.model_path}")
    for label, geom_names in result.support_geom_groups:
        print(f"[INFO]   {label}: {', '.join(geom_names)}")


if __name__ == "__main__":
    main()
