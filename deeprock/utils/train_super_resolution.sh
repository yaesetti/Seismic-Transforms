echo "----------------------------------------------------"
echo "Seismic Super-Resolution Training"

# Notebook-based default training run
python train.py --model deeplabv3 --task super_resolution --dataset seismic_super_resolution --root_data_dir /workspaces/seismic-datasets/data/processed/super_resolution/seismic_super_resolution/partition_method_random --label_subdir label --step train-evaluate --accelerator gpu --devices "[0]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_random --run_id 0 --precision "32-true" --batch_size 64
# ImageNet-V1
python train.py --model deeplabv3 --task super_resolution --dataset seismic_super_resolution --root_data_dir /workspaces/seismic-datasets/data/processed/super_resolution/seismic_super_resolution/partition_method_random --label_subdir label --step train-evaluate --accelerator gpu --devices "[0]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet --run_id 0 --precision "32-true" --pretrained --batch_size 64
# ImageNet-V1 (freeze)
python train.py --model deeplabv3 --task super_resolution --dataset seismic_super_resolution --root_data_dir /workspaces/seismic-datasets/data/processed/super_resolution/seismic_super_resolution/partition_method_random --label_subdir label --step train-evaluate --accelerator gpu --devices "[0]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet_freeze --run_id 0 --precision "32-true" --pretrained --freeze_backbone --batch_size 64

echo "----------------------------------------------------"