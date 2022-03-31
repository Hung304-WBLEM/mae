# python submitit_pretrain_custom_v3_linformer.py \
#     --job_dir jobdir/highres_v3_linformer-k128_vit_base_patch16_input112_combined_datasets \
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

python -m torch.distributed.launch --master_port=6667 --nproc_per_node=4 main_pretrain_custom_v3_performer.py \
    --output_dir jobdir/highres_v3_performer_vit_base_patch16_input112_combined_datasets \
    --log_dir jobdir/highres_v3_performer_vit_base_patch16_input112_combined_datasets \
    --num_workers 8\
    --batch_size 256 \
    --model mae_vit_base_patch16 \
    --input_size 112 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --dataset combined_datasets
