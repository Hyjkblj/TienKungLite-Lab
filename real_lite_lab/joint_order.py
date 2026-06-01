from __future__ import annotations


def build_target_order_indices(source_order: list[str] | tuple[str, ...], target_order: list[str] | tuple[str, ...]) -> list[int]:
    if len(source_order) != len(target_order):
        raise ValueError(f"Order length mismatch: {len(source_order)} vs {len(target_order)}.")

    if set(source_order) != set(target_order):
        raise ValueError(f"Order names do not match.\nSource: {list(source_order)}\nTarget: {list(target_order)}")

    target_name_to_idx = {name: idx for idx, name in enumerate(target_order)}
    return [target_name_to_idx[name] for name in source_order]
