# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from hydra import initialize, compose
from omegaconf import OmegaConf
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args, hydra_overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=hydra_overrides)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    trainer = Trainer(**cfg, resolved_config=resolved_cfg)
    trainer.run()


if __name__ == "__main__":
    main()


