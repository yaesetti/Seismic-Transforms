from pathlib import Path, PosixPath
import sys
import yaml
import pandas as pd
from typing import Any, Dict, List, Optional

from deeprock.main import benchmark
from jsonargparse import auto_cli, capture_parser, strip_meta


def path_representer(dumper, data):
    return dumper.represent_str(str(data))


yaml.add_representer(Path, path_representer)
yaml.add_representer(PosixPath, path_representer)


def get_parser():
    return capture_parser(
        lambda: auto_cli(benchmark, as_positional=False, fail_untyped=True)
    )


def main(args: Optional[List[Any]] = None):
    parser = get_parser()
    args = parser.parse_args(args=args, with_meta=False)
    args = strip_meta(args)
    objects = parser.instantiate_classes(args).as_dict()
    objects.pop("config", None)  # Remove config if it exists

    if "train" in objects["stages"]:
        args.pop("config", None)  # Remove config from args if it exists
        objects["experiment_metadata"] = parser.dump(args, format="parser_mode")

    benchmark(**objects)

if __name__ == "__main__":
    main()
    # main(args=[
    #     "--backbone", "class_path: minerva.models.nets.image.deeplabv3.DeepLabV3Backbone",
    #     "--task", "facies:aicrowd_seamai",
    #     "--data_dir", "data/",
    #     "--logs_dir", "logs_teste",
    #     "--prediction_head", "linear",
    #     "--epochs", "1",
    #     "--learning_rate", "1e-3",
    #     "--freeze_backbone", "True",
    #     "--norm", "znorm",
    #     "--accelerator", "gpu",
    #     "--devices", "1",
    #     "--batch_size", "16",
    # ])
