python maeseg_target.py domain0.75_msd_lr7.5e-4_loadrecon_1.0recon_1.0pseudo \
    -G $1 \
    --method domain_adaptation \
    --train_list MSD_train \
    --val_list MSD_val \
    --batch_size 1 \
    --data_root /HDD_data/MING/VAE/MSD/Task07_Pancreas/data \
    --val_data_root /HDD_data/MING/VAE/MSD/Task07_Pancreas/data \
    --data_path /home/user02/TUTMING/ming/VAE/data/My_Multi_all.json \
    --eval_epoch 1 \
    --save_epoch 400 \
    --max_epoch 2400 \
    --save_more_reference \
    --load_prefix domain0.75_msd_lr7.5e-4_loadrecon_1.0recon \
    --load_prefix_mae mae_nih_pmask0.75_img128p16_batch1_all_rotation \
    --lr_mae 7.5e-4 \
