import os
import argparse

import torch
import torch.nn as nn
import timm.models.vision_transformer

from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone

from deeprock.main import benchmark

task_images = {
    "facies:aicrowd_seamai": (1008, 592),
    # "binary_channel_identification:cigChannel": (256, 256),
    # "channel_identification:cigChannel": (256, 256),
    # "faults:thebe": (96, 96),
    # "salt:tgs": (112, 112),
    # "seismic_attribute_computation:F3:envelope": (256, 704),
    # "seismic_attribute_computation:F3:semblance": (256, 704),
    # "super_resolution:seismic_super_resolution": (128, 128),
}

def instantiate(random=False):
    model = DeepLabV3Backbone(pretrained=False)
    ckpt = '/petrobr/parceirosbr/spfm/pretrain_weights/resnet50/deeplabv3_resnet50_coco-cd0a2569.pth'
    if not random:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)

        # Keep only backbone weights and strip the "backbone." prefix
        backbone_state_dict = {
            k[len("backbone."):]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.") and not k.startswith("backbone.fc.")
        }

        missing, unexpected = model.RN50model.load_state_dict(
            backbone_state_dict,
            strict=False,
        )
        print("Loaded model")
        print("Missing:", missing)
        print("Unexpected:", unexpected)

    return model

def adapter(
    backbone: nn.Module, input_data: torch.Tensor, task_type: str
) -> torch.Tensor:
    x = backbone(input_data)
    if task_type == "classification":
        pass
    else: # regression and segmentation use patch tokens and not CLS tokens
        pass
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction-head",
        "-p",
        choices=["linear", "mlp", "aspp", "linear-regression", "mlp-regression"],
        default="linear-regression"
    )

    parser.add_argument(
        "--random",
        "-r",
        action="store_true"
    )

    parser.add_argument(
        "--freeze",
        "-f",
        action="store_true"
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--tasks',
        nargs="+",
        required=False,
        default=None
    )

    parser.add_argument(
        "--lr",
        default=1e-4,
        required=False,
        type=float
    )

    parser.add_argument(
        "--stages",
        default="train+evaluate",
        required=False,
        choices=["train", "evaluate", "train+evaluate"]
    )

    args = parser.parse_args()

    lr_str = "" if args.lr == 1e-4 else f"_lr={args.lr}"

    output_prj_path = "/petrobr/parceirosbr/spfm/victor.setti/outputs/"
    logs_dir = os.path.join(
        output_prj_path,
        f"deeplab_{args.prediction_head}_{'random' if args.random else 'pretrain'}_{'freeze' if args.freeze else 'fullft'}{lr_str}",
    )

    tasks_to_run = args.tasks
    if tasks_to_run is None:
        tasks_to_run = list(task_images.keys())

    print(f"The following tasks will be executed (in order): {tasks_to_run}")
    for task in tasks_to_run:
        base_task = task.split('+')[0]

        img_size = task_images[base_task]

        try:
            print(f"**Running {task} with DeepLabV3 + {args.prediction_head} head \n\tfreeze_backbone: {args.freeze}\n\tseed: {args.seed}\n\trandom: {args.random}\n\tLog dir {logs_dir}")
            backbone = instantiate(
                random=args.random,
            )

            if task in ["seismic_attribute_computation:F3:envelope", "seismic_attribute_computation:F3:semblance", "super_resolution:seismic_super_resolution"]:
                norm=None
            else:
                norm="znorm"

            benchmark(
                backbone=backbone,
                task=task,
                data_dir="/petrobr/parceirosbr/spfm/datasets/deeprock_datasets/",
                logs_dir=logs_dir,
                prediction_head=args.prediction_head.replace("-regression", ""),
                epochs = 100,
                learning_rate = args.lr,
                optimizer= torch.optim.Adam,
                model_resolution=16, # is it 8 or 16? VS: Was 8, changed to 16
                channels=3,
                accelerator="gpu",
                freeze_backbone=args.freeze,
                norm=norm, # ?
                enforce_interpolation=False if "regression" in args.prediction_head else True, # use small prediction head (smaller output) and interpolate image to original size (input)
                execution_id=args.seed,
                progress_bar_refresh_rate=50,
                skip_completed=False,
                stages=args.stages,
                # TODO: Change this path
                # bsds_path="/tatu/unicamp_spfm/deeprock_exps/seismic-datasets/.cache/an_pybsds500/CNNforFaultInterpretation/py-bsds500",
            )

            print("-" * 80)
            print()
        except Exception as e:
            print(f"Error executing experiment to log_dir: {logs_dir} (task={task})")
            print(e)
            print("-" * 80)
            print()

    print("All jobs evaluated!")
