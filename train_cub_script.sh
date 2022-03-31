# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain.py \
#        --output_dir jobdir/vit_base_patch16_input112_cub_200_2011 \
#        --log_dir jobdir/vit_base_patch16_input112_cub_200_2011 \
#        --num_workers 8\
#        --batch_size 256 \
#        --model mae_vit_base_patch16 \
#        --input_size 112 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 500 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset cub_200_2011

# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain_custom_v2.py \
#        --output_dir jobdir/highres_v2_vit_base_patch16_input112_cub_200_2011 \
#        --log_dir jobdir/highres_v2_vit_base_patch16_input112_cub_200_2011 \
#        --num_workers 8\
#        --batch_size 256 \
#        --model mae_vit_base_patch16 \
#        --input_size 112 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 500 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset cub_200_2011
# 
# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain_custom_v3_nystromer.py \
#        --output_dir jobdir/highres_v3_nystromer_vit_base_patch16_input112_cub_200_2011 \
#        --log_dir jobdir/highres_v3_nystromer_vit_base_patch16_input112_cub_200_2011 \
#        --num_workers 8\
#        --batch_size 256 \
#        --model mae_vit_base_patch16 \
#        --input_size 112 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 500 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset cub_200_2011
# 

# FROM PRETRAINED IMAGENET SSL CHECKPOINT
# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain.py \
#        --output_dir jobdir/vit_base_patch16_input224_cub_200_2011_from_pretrained \
#        --log_dir jobdir/vit_base_patch16_input224_cub_200_2011_from_pretrained \
#        --num_workers 8\
#        --batch_size 64 \
#        --model mae_vit_base_patch16 \
##       --resume '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/pretrained_models/mae_pretrain_vit_base_full.pth' \
#        --resume '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/vit_base_patch16_input224_cub_200_2011_from_pretrained/checkpoint-400.pth' \
#        --input_size 224 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 1000 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset cub_200_2011



# python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain_custom_v3_nystromer.py \
#        --output_dir jobdir/highres_v3_nystromer_vit_base_patch16_input224_cub_200_2011_from_pretrained \
#        --log_dir jobdir/highres_v3_nystromer_vit_base_patch16_input224_cub_200_2011_from_pretrained \
#        --num_workers 8\
#        --batch_size 64 \
#        --model mae_vit_base_patch16 \
#        --resume '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/highres_v3_nystromer_vit_base_patch16_input224_cub_200_2011_from_pretrained/checkpoint-700.pth' \
#        --input_size 224 \
#        --norm_pix_loss \
#        --mask_ratio 0.75 \
#        --epochs 1000 \
#        --warmup_epochs 40 \
#        --blr 1.5e-4 --weight_decay 0.05 \
#        --dataset cub_200_2011


# Cropped Version
python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain.py \
       --output_dir jobdir/vit_base_patch16_input224_crop_cub_200_2011_from_pretrained \
       --log_dir jobdir/vit_base_patch16_input224_crop_cub_200_2011_from_pretrained \
       --num_workers 8\
       --batch_size 64 \
       --model mae_vit_base_patch16 \
       --resume '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/pretrained_models/mae_pretrain_vit_base_full.pth' \
       --input_size 224 \
       --norm_pix_loss \
       --mask_ratio 0.75 \
       --epochs 500 \
       --warmup_epochs 40 \
       --blr 1.5e-4 --weight_decay 0.05 \
       --dataset cub_200_2011

python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_pretrain_custom_v3_nystromer.py \
       --output_dir jobdir/highres_v3_nystromer_vit_base_patch16_input224_crop_cub_200_2011_from_pretrained \
       --log_dir jobdir/highres_v3_nystromer_vit_base_patch16_input224_crop_cub_200_2011_from_pretrained \
       --num_workers 8\
       --batch_size 64 \
       --model mae_vit_base_patch16 \
       --resume '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/pretrained_models/mae_pretrain_vit_base_full.pth' \
       --input_size 224 \
       --norm_pix_loss \
       --mask_ratio 0.75 \
       --epochs 500 \
       --warmup_epochs 40 \
       --blr 1.5e-4 --weight_decay 0.05 \
       --dataset cub_200_2011

