python submitit_pretrain.py \
    --job_dir jobdir/highres_vit_base_patch16_input224_five_classes_mass_calc_pathology \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 30 \
    --model mae_vit_base_patch16 \
    --input_size 224 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --partition 'batch' \
    --dataset five_classes_mass_calc_pathology

python submitit_pretrain.py \
    --job_dir jobdir/highres_vit_base_patch16_input224_combined_datasets \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 30 \
    --model mae_vit_base_patch16 \
    --input_size 224 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --partition 'batch' \
    --dataset combined_datasets

python submitit_pretrain.py \
    --job_dir jobdir/highres_vit_base_patch16_input224_aug_combined_datasets \
    --ngpus 4 \
    --nodes 1 \
    --timeout 17280 \
    --batch_size 30 \
    --model mae_vit_base_patch16 \
    --input_size 224 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --partition 'batch' \
    --dataset aug_combined_datasets
