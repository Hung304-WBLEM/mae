python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4  main_finetune.py \
    --batch_size 256 \
    --model vit_base_patch16 \
    --finetune '/home/hqvo2/Projects/Breast_Cancer/libs/mae/jobdir/pretrained_models/mae_pretrain_vit_base.pth'\
    --epochs 10 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --dataset cub_200_2011
