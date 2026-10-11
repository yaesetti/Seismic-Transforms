
echo "----------------------------------------------------"
echo "Salt Segmentation Training"
# Random
python train.py --model deeplabv3 --task salt --dataset tgs --root_data_dir /workspaces/seismic-datasets/data/processed/salt/tgs/partition_method_random --accelerator gpu --devices "[1]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_random --run_id 0 --precision "32-true" --dice_weight 0.5 --bce_weight 0.5 --batch_size 64
# # ImageNet-V1
python train.py --model deeplabv3 --task salt --dataset tgs --root_data_dir /workspaces/seismic-datasets/data/processed/salt/tgs/partition_method_random --accelerator gpu --devices "[1]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet --run_id 0 --precision "32-true" --pretrained --dice_weight 0.5 --bce_weight 0.5 --batch_size 64
# # ImageNet-V1 (freeze)
python train.py --model deeplabv3 --task salt --dataset tgs --root_data_dir /workspaces/seismic-datasets/data/processed/salt/tgs/partition_method_random --accelerator gpu --devices "[1]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet_freeze --run_id 0 --precision "32-true" --pretrained  --freeze_backbone --dice_weight 0.5 --bce_weight 0.5 --batch_size 64
echo "----------------------------------------------------"


