# python submitit_pretrain.py \
#     --job_dir jobdir/vit_base_patch16_input64_mask0.5_five_classes_mass_calc_pathology \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 17280 \
#     --batch_size 128 \
#     --model mae_vit_base_patch16 \
#     --input_size 64 \
#     --norm_pix_loss \
#     --mask_ratio 0.5 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset five_classes_mass_calc_pathology \


# python submitit_pretrain.py \
#     --job_dir jobdir/vit_base_patch16_input64_mask0.5_combined_datasets \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 17280 \
#     --batch_size 128 \
#     --model mae_vit_base_patch16 \
#     --input_size 64 \
#     --norm_pix_loss \
#     --mask_ratio 0.5 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset combined_datasets \

# python submitit_pretrain.py \
#     --job_dir jobdir/vit_large_patch16_input64_mask0.75_five_classes_mass_calc_pathology \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 17280 \
#     --batch_size 128 \
#     --model mae_vit_large_patch16 \
#     --input_size 64 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset five_classes_mass_calc_pathology \


# python submitit_pretrain.py \
#     --job_dir jobdir/vit_large_patch16_input64_mask0.75_combined_datasets \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 17280 \
#     --batch_size 128 \
#     --model mae_vit_large_patch16 \
#     --input_size 64 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset combined_datasets \

# python submitit_pretrain.py \
#     --job_dir jobdir/vit_base_patch16_e500_input224_image_lesion_combined_datasets \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 17280 \
#     --batch_size 128 \
#     --model mae_vit_base_patch16 \
#     --input_size 224 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset image_lesion_combined_datasets \

# python submitit_pretrain.py \
#     --job_dir jobdir/vit_base_patch16_e500_wd0.01_input224_image_lesion_combined_datasets \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 17280 \
#     --batch_size 128 \
#     --model mae_vit_base_patch16 \
#     --input_size 64 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.01 \
#     --partition 'batch' \
#     --dataset image_lesion_combined_datasets \

# python submitit_pretrain.py \
#     --job_dir jobdir/vit_base_patch16_e500_blr1.5e-3_input224_image_lesion_combined_datasets \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 17280 \
#     --batch_size 128 \
#     --model mae_vit_base_patch16 \
#     --input_size 64 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-3 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset image_lesion_combined_datasets \

# python submitit_pretrain.py \
#     --job_dir jobdir/vit_base_patch16_e500_input224_aug_combined_datasets \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 17280 \
#     --batch_size 128 \
#     --model mae_vit_base_patch16 \
#     --input_size 224 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset aug_combined_datasets \

# python -m torch.distributed.launch --master_port=6667 --nproc_per_node=4 main_pretrain.py \
#     --output_dir jobdir/vit_base_patch16_input224_b64_combined_datasets \
#     --log_dir jobdir/vit_base_patch16_input224_b64_combined_datasets \
#     --num_workers 16\
#     --batch_size 64 \
#     --model mae_vit_base_patch16 \
#     --input_size 224 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --dataset combined_datasets \


python -m torch.distributed.launch --master_port=6667 --nproc_per_node=4 main_pretrain.py \
    --output_dir jobdir/nomaskboth_visualize_vit_base_patch16_input112_combined_datasets \
    --log_dir jobdir/nomaskboth_visualize_vit_base_patch16_input112_combined_datasets \
    --num_workers 10\
    --batch_size 256 \
    --model mae_vit_base_patch16 \
    --input_size 112 \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --dataset combined_datasets \
