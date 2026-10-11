echo "----------------------------------------------------"
echo "Seismic Channel Identification Training"

# Notebook-based default training run
python train.py --model deeplabv3 --task channel_identification --dataset cigChannel --root_data_dir /workspaces/seismic-datasets/data/processed/channel_identification/cigChannel/partition_method_random --step train-evaluate --accelerator gpu --devices "[0]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_random --run_id 0 --precision "16-mixed" --batch_size 64
# ImageNet-V1
python train.py --model deeplabv3 --task channel_identification --dataset cigChannel --root_data_dir /workspaces/seismic-datasets/data/processed/channel_identification/cigChannel/partition_method_random --step train-evaluate --accelerator gpu --devices "[0]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet --run_id 0 --precision "16-mixed" --pretrained --batch_size 64
# ImageNet-V1 (freeze)
python train.py --model deeplabv3 --task channel_identification --dataset cigChannel --root_data_dir /workspaces/seismic-datasets/data/processed/channel_identification/cigChannel/partition_method_random --step train-evaluate --accelerator gpu --devices "[0]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet_freeze --run_id 0 --precision "16-mixed" --pretrained --freeze_backbone --batch_size 64

echo "----------------------------------------------------"