
echo "----------------------------------------------------"
echo "Seismic Facies Classification Training"
# Seismic Facies Classification Training Script
# Random
python train.py --model deeplabv3 --task facies --dataset aicrowd_seamai --root_data_dir /workspaces/seismic-datasets/data/processed/facies/aicrowd_seamai/partition_method_test_first --accelerator gpu --devices "[1]" --max_epochs 100 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_random --run_id 0 --precision "16-mixed" --batch_size 8
# # ImageNet-V1
python train.py --model deeplabv3 --task facies --dataset aicrowd_seamai --root_data_dir /workspaces/seismic-datasets/data/processed/facies/aicrowd_seamai/partition_method_test_first --accelerator gpu --devices "[1]" --max_epochs 100 --pretrained --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet --run_id 0 --precision "16-mixed" --batch_size 8
# ImageNet-V1 (freeze)
python train.py --model deeplabv3 --task facies --dataset aicrowd_seamai --root_data_dir /workspaces/seismic-datasets/data/processed/facies/aicrowd_seamai/partition_method_test_first --accelerator gpu --devices "[1]" --max_epochs 100 --pretrained --freeze_backbone --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet_freeze --run_id 0 --precision "16-mixed" --batch_size 8
echo "----------------------------------------------------"


