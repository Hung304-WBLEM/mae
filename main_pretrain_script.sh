python -m torch.distributed.launch --nproc_per_node=2 main_pretrain_custom_independent_patches.py \
       --batch_size 128 \
       --model mae_vit_base_patch16 \
       --input_size 112 \
       --norm_pix_loss \
       --mask_ratio 0.75 \
       --epochs 800 \
       --warmup_epochs 40 \
       --blr 1.5e-4 --weight_decay 0.05 \
       --dataset five_classes_mass_calc_pathology
