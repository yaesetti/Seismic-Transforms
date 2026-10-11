
echo "----------------------------------------------------"
echo "Fault Segmentation Training"
# Random
python train.py --model deeplabv3 --task faults --dataset thebe --root_data_dir /workspaces/seismic-datasets/data/processed/faults/thebe/partition_method_predefined --accelerator gpu --devices "[1]" --max_epochs 1 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_random --run_id 0 --precision "16-mixed" --batch_size 128
# # # ImageNet-V1
# python train.py --model deeplabv3 --task faults --dataset thebe --root_data_dir /workspaces/seismic-datasets/data/processed/faults/thebe/partition_method_predefined --accelerator gpu --devices "[1]" --max_epochs 1 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet --run_id 0 --precision "16-mixed" --batch_size 128 --pretrained
# # # ImageNet-V1 (freeze)
# python train.py --model deeplabv3 --task faults --dataset thebe --root_data_dir /workspaces/seismic-datasets/data/processed/faults/thebe/partition_method_predefined --accelerator gpu --devices "[1]" --max_epochs 1 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet_freeze --run_id 0 --precision "16-mixed" --batch_size 128 --pretrained --freeze_backbone
echo "----------------------------------------------------"


