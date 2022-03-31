# python submitit_pretrain_custom_v2.py \
#     --job_dir jobdir/highres_v2_nomaskboth_vit_base_patch16_input112_combined_datasets \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 600 \
#     --batch_size 256 \
#     --model mae_vit_base_patch16 \
#     --input_size 112 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset combined_datasets

# python -m torch.distributed.launch --master_port=6667 --nproc_per_node=4 main_pretrain_custom_v2.py \
#     --output_dir jobdir/highres_v2_nomaskboth_vit_base_patch16_input112_combined_datasets \
#     --log_dir jobdir/highres_v2_nomaskboth_vit_base_patch16_input112_combined_datasets \
#     --num_workers 8\
#     --batch_size 256 \
#     --model mae_vit_base_patch16 \
#     --input_size 112 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --dataset combined_datasets

# python submitit_pretrain_custom_v2.py \
#     --job_dir jobdir/highres_v2_vit_base_patch16_input112_cub_200_2011 \
#     --ngpus 4 \
#     --nodes 1 \
#     --timeout 600 \
#     --batch_size 256 \
#     --model mae_vit_base_patch16 \
#     --input_size 112 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 500 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --partition 'batch' \
#     --dataset cub_200_2011
 
python -m torch.distributed.launch --master_port=6667 --nproc_per_node=4 main_pretrain_custom_v2.py \
    --output_dir jobdir/test \
    --log_dir jobdir/test \
    --num_workers 10\
    --batch_size 16 \
    --model mae_vit_base_patch16 \
    --input_size 224 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --dataset combined_datasets
