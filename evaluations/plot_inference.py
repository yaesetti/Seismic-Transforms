import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# ------------ Custom Minerva/Seismic Imports ------------
sys.path.append('/petrobr/parceirosbr/home/victor.setti/workspace/Seismic-Transforms')

from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone, DeepLabV3PredictionHead
from minerva.transforms.transform import TransformPipeline, Transpose, Repeat, CastTo
from minerva.data.readers.numpy_reader import NumpyFolderReader
from minerva.data.datasets.base import SimpleDataset

from seismic.seismic_model import SeismicModel
from seismic.binary_segmentation_loss import BinarySegmentationLoss
from seismic.padding import SafePadding
from seismic.transformed_reader import TransformedReader

def main():
    parser = argparse.ArgumentParser(description="Plot side-by-side inference of a specific TGS sample.")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment directory")
    parser.add_argument("--group-name", type=str, default=None, help="Name of the parent grouping directory")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the test sample to visualize (e.g., 0 to 399)")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Setup Paths
    # ---------------------------------------------------------
    OUT_PROJECT_ROOT = Path("/petrobr/parceirosbr/spfm/victor.setti/outputs/tgs/")
    OUT_PERSONAL_ROOT = Path("/petrobr/parceirosbr/home/victor.setti/workspace/Seismic-Transforms/outputs/tgs/")
    DATASET_ROOT = Path("/petrobr/parceirosbr/spfm/datasets/seismic-datasets/data/tasks/salt_body_segmentation/tgs/processed_data/partition_method_random")

    if args.group_name:
        ckpt_dir = OUT_PROJECT_ROOT / args.group_name / args.exp_name / "checkpoints"
        plots_dir = OUT_PERSONAL_ROOT / args.group_name / args.exp_name / "plots"
    else:
        ckpt_dir = OUT_PROJECT_ROOT / args.exp_name / "checkpoints"
        plots_dir = OUT_PERSONAL_ROOT / args.exp_name / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.ckpt"
    
    if not ckpt_path.exists():
        print(f"❌ Error: Could not find checkpoint at {ckpt_path}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Load the Dataset Pipeline
    # ---------------------------------------------------------
    IGNORE_INDEX = 255

    data_transform_pipeline = TransformPipeline([
        SafePadding(128, 128, padding_mode='reflect'),
        Transpose([2, 0, 1]),
        Repeat(axis=0, n_repetitions=3)
    ])

    label_transform_pipeline = TransformPipeline([
        SafePadding(128, 128, padding_mode='constant', padding_value=IGNORE_INDEX, mask_padding_value=IGNORE_INDEX),
        Transpose([2, 0, 1]),
        CastTo(np.float32)
    ])

    test_dataset = SimpleDataset(
        readers=[
            TransformedReader(reader=NumpyFolderReader(path=DATASET_ROOT / "test" / "data", allow_pickle=True), transform=data_transform_pipeline),
            TransformedReader(reader=NumpyFolderReader(path=DATASET_ROOT / "test" / "label", allow_pickle=True), transform=label_transform_pipeline)
        ]
    )

    if args.sample_idx >= len(test_dataset):
        print(f"❌ Error: sample-idx {args.sample_idx} is out of bounds for test dataset of size {len(test_dataset)}")
        sys.exit(1)

    # Get the specific image and mask
    img, mask = test_dataset[args.sample_idx]

    # ---------------------------------------------------------
    # 3. Load the Model and Checkpoint
    # ---------------------------------------------------------
    print("Loading model architecture and weights...")
    deeplab_backbone = DeepLabV3Backbone(num_classes=1)
    pred_head = DeepLabV3PredictionHead(num_classes=1)

    model = SeismicModel(
        backbone=deeplab_backbone,
        pred_head=pred_head,
        num_classes=1,
        loss_fn=BinarySegmentationLoss(ignore_index=IGNORE_INDEX)
    )

    # Added weights_only=False to silence the PyTorch security warning
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # ---------------------------------------------------------
    # 4. Run Inference
    # ---------------------------------------------------------
    with torch.no_grad():
        # Add batch dimension: shape becomes (1, C, H, W)
        img_tensor = torch.tensor(img).unsqueeze(0)
        
        # --- THE FIX ---
        # Minerva asserts batch_size > 1. We duplicate the image to create a batch of 2.
        img_tensor = img_tensor.repeat(2, 1, 1, 1)
        
        # Forward pass (Batch Size = 2)
        logits = model(img_tensor)
        
        # Keep only the first prediction and discard the duplicate
        logits = logits[0:1]
        
        # Apply sigmoid and threshold at 0.5 to get binary predictions
        pred_prob = torch.sigmoid(logits).squeeze().numpy()
        pred_mask = (pred_prob > 0.5).astype(np.float32)

    # ---------------------------------------------------------
    # 5. Plotting
    # ---------------------------------------------------------
    plt.style.use('dark_background') # Better for seismic contrast
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # The image was repeated 3 times across channels (C, H, W). We just need one channel.
    img_viz = img[0, :, :]
    mask_viz = mask[0, :, :]

    # --- THE FIX: CROP OUT THE PADDING ---
    # Find the coordinates where the mask does NOT have the padding value (IGNORE_INDEX)
    valid_y, valid_x = np.where(mask_viz != IGNORE_INDEX)

    if len(valid_y) > 0 and len(valid_x) > 0:
        y_min, y_max = valid_y.min(), valid_y.max() + 1
        x_min, x_max = valid_x.min(), valid_x.max() + 1
        
        # Crop the padding out of all three arrays
        img_viz = img_viz[y_min:y_max, x_min:x_max]
        mask_viz = mask_viz[y_min:y_max, x_min:x_max]
        pred_mask = pred_mask[y_min:y_max, x_min:x_max]

    # Clean up any remaining ignored values (just in case)
    mask_viz = np.where(mask_viz == IGNORE_INDEX, 0, mask_viz)

    # 1. Original Image
    axes[0].imshow(img_viz, cmap='gray')
    axes[0].set_title(f"Seismic Image (Test Index: {args.sample_idx})", fontsize=14)
    axes[0].axis('off')

    # 2. Ground Truth
    axes[1].imshow(mask_viz, cmap='inferno')
    axes[1].set_title("Ground Truth Mask", fontsize=14)
    axes[1].axis('off')

    # 3. Model Prediction
    axes[2].imshow(pred_mask, cmap='inferno')
    axes[2].set_title("Model Prediction", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    
    plot_filename = plots_dir / f"inference_sample_{args.sample_idx}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved side-by-side inference plot to: {plot_filename}")

if __name__ == "__main__":
    main()