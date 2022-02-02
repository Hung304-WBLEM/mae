python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
       --batch_size 30 \
       --model mae_vit_base_patch16 \
       --input_size 224 \
       --norm_pix_loss \
       --mask_ratio 0.75 \
       --epochs 800 \
       --warmup_epochs 40 \
       --blr 1.5e-4 --weight_decay 0.05 \
       --dataset five_classes_mass_calc_pathology \
