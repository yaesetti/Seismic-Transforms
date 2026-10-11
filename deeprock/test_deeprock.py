from deeprock.main import benchmark
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone
from pathlib import Path
import torch

available_tasks = [         # Check deeprock.main.TASKS
    "binary_channel_identification:cigChannel",
    "facies:aicrowd_seamai",
    "faults:thebe",
    "salt:tgs",
    "seismic_attribute_computation:F3:envelope",
    "seismic_attribute_computation:F3:semblance",
    "super_resolution:seismic_super_resolution"
]

prediction_heads = [        # Check deeprock.main.PREDICTION_HEADS
    "linear",
    "mlp",
    "aspp",
    "pup",
]

def main():
    backbone = DeepLabV3Backbone(pretrained=False)
    task = "salt:tgs"
    data_dir = Path("/home/shared/deeprock_datasets/")
    logs_dir = Path("./logs_teste")

    # The model is composed of backbone + adapter + prediction head. The adapter is optional and can be used to adapt the backbone features to the prediction head. If no adapter is provided, the backbone features are directly fed to the prediction head (torch.nn.Identity).
    # Thus, a forward involves: y_hat = prediction_head(backbone_adapter(backbone(x)))
    benchmark(
        backbone=backbone,
        task=task,                      # The task to benchmark
        data_dir=data_dir,              # Root of deeprock datasets (should contain the folders for each dataset)
        logs_dir=logs_dir,              # Where to save the logs and checkpoints
        prediction_head="aspp",         # Can be a string or a torch.nn.Module
        backbone_adapter=None,          # A torch.nn.Module that adapts the backbone features to the prediction head. If None, the backbone features are directly fed to the prediction head (torch.nn.Identity).
        epochs=100,                     # Number of epochs to train the model
        batch_size=256,
        learning_rate=(1e-3, 1e-4),     # Learning rate for the optimizer
        optimizer=torch.optim.AdamW,    # Optimizer class to use for training
        model_resolution=16,            # Resolution of the model   
        channels=3,                     # Number of channels that model expects as input (SFM=1, DLv3=3)
        freeze_backbone=False,          # Whether to freeze the backbone during training (True or False)
        norm="znorm",                   # Normalization to apply to the input data. It is applyed per-instance
        accelerator="gpu",              # Accelerator to use for training (cpu, gpu, tpu, etc.)
        devices=1,                      # Numeber of accelerators to use for training (1, 2, 4, etc. GPUS)
        progress_bar_refresh_rate=50,   # How often to refresh the progress bar (in steps)
        stages="train+evaluate",        # Control the training process. Can be "train", "evaluate" or "train+evaluate".
        skip_completed=False,           # Whether to skip the training/evaluation if the model has already been trained and evaluated (True or False)  
        execution_id=1
    )
    

if __name__ == "__main__":
    main()