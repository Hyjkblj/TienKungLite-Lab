from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


REAL_LITE_GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(-0.02, 0.04),
            noise_step=0.02,
            border_width=0.25,
        )
    },
)
