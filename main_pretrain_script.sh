# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain.py \
#        --batch_size 256 \
#        --model mae_vit_base_patch16 \
#        --input_size 112 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 800 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset cub_200_2011

# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain_custom_v2.py \
#        --batch_size 256 \
#        --model mae_vit_base_patch16 \
#        --input_size 112 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 800 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset cub_200_2011

# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain_custom_v3_nystromer.py \
#        --batch_size 64 \
#        --model mae_vit_base_patch16 \
#        --input_size 224 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 800 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset combined_datasets

# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain_custom_v2_3branches.py \
#        --batch_size 64 \
#        --model mae_vit_base_patch16 \
#        --input_size 112 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 800 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset combined_datasets

python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain_custom_swin.py \
       --batch_size 1 \
       --model mae_swin_base_patch4 \
       --input_size 112 \
       --norm_pix_loss \
       --mask_ratio 0.75 \
       --epochs 800 \
       --warmup_epochs 40 \
       --blr 1.5e-4 --weight_decay 0.05 \
       --dataset combined_datasets
