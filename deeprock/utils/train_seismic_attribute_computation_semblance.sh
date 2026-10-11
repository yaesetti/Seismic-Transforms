echo "----------------------------------------------------"
echo "Seismic Attribute Computation Training"

# Random
python train.py --model deeplabv3 --task seismic_attribute_calculation --dataset F3 --root_data_dir /workspaces/seismic-datasets/data/processed/seismic_attribute_calculation/F3/partition_method_test_predefined_val_last --label_subdir attributes/semblance --accelerator gpu --devices "[1]" --max_epochs 1 --enable_checkpointing --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_random --run_id 0 --precision "32-true" --batch_size 16
# # ImageNet-V1
# python train.py --model deeplabv3 --task seismic_attribute_calculation --dataset F3 --root_data_dir /workspaces/seismic-datasets/data/processed/seismic_attribute_calculation/F3/partition_method_test_predefined_val_last --label_subdir attributes/semblance --accelerator gpu --devices "[1]" --max_epochs 1 --pretrained --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet --run_id 0 --precision "32-true" --pretrained --batch_size 16
# # ImageNet-V1 (freeze)
# python train.py --model deeplabv3 --task seismic_attribute_calculation --dataset F3 --root_data_dir /workspaces/seismic-datasets/data/processed/seismic_attribute_calculation/F3/partition_method_test_predefined_val_last --label_subdir attributes/semblance --accelerator gpu --devices "[1]" --max_epochs 1 --pretrained --progress_bar_refresh_rate 10 --root_log_dir logs/dlv3_imagenet_freeze --run_id 0 --precision "32-true" --pretrained --freeze_backbone --batch_size 16
echo "----------------------------------------------------"