python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_finetune.py --eval \
    --resume '/home/hqvo2/Projects/Breast_Cancer/libs/mae/finetune_jobdir/pretrained_vit_base_patch16_input224_lr5e-4_crop_cub_200_2011/checkpoint-99.pth' \
    --model vit_base_patch16 --batch_size 256 --dataset cub_200_2011 --nb_classes 200

python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_finetune.py --eval \
    --resume '/home/hqvo2/Projects/Breast_Cancer/libs/mae/finetune_jobdir/vit_base_patch16_input224_lr5e-4_crop_cub_200_2011_from_pretrained/checkpoint-84.pth' \
    --model vit_base_patch16 --batch_size 256 --dataset cub_200_2011 --nb_classes 200

python -m torch.distributed.launch --master_port=6666 --nproc_per_node=4 main_finetune.py --eval \
    --resume '/home/hqvo2/Projects/Breast_Cancer/libs/mae/finetune_jobdir/highres_v3_nystromer_vit_base_patch16_input224_lr5e-4_crop_cub_200_2011_from_pretrained/checkpoint-86.pth' \
    --model vit_base_patch16 --batch_size 256 --dataset cub_200_2011 --nb_classes 200
