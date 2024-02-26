python3 data_preparation_av.py \
 --dataset_root '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' \
 --output_dir './dataset_list/ucf_10per_paramshakti/' \
 --trainlist_path '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/ucf101_train_split_1_videos.txt' \
 --testlist_path '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/ucf101_val_split_1_videos.txt' \
 --percentage 10



python3 main.py \
 --data_dir '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' \
 --list_root './dataset_list/ucf_10per_paramshakti/' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 50 --sched cosine --duration 8 --batch-size 48 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/test/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained



//For revswin ckpt noflip epoch 100 wd 0.3
//gnode4 screen revswin_scratch
python3 main.py --data_dir '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' --list_root './dataset_list/ucf_10per_paramshakti/' --use_pyav --dataset 'ucf101'  --opt adamw --lr 1e-3 --epochs 100 --sched cosine --duration 8 --batch-size 48  --super_img_rows 3 --disable_scaleup  --mixup 0.8 --cutmix 1.0 --drop-path 0.1    --model sifar_base_patch4_window12_192_3x3  --output_dir './output/revswin_ckpt_norm_only_noflip_sup_epoch_100_10per_ucf_bs48_gamma0.6_beta1_lr2e-6_temp0.5_wd_0.3/' --hpe_to_token  --sup_thresh 100 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5  --gamma 0.6 --beta 1 --pretrained --weight-decay 0.3



// gpu007 login04
// screen revswin
// no flip
python3 main.py \
 --data_dir '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' \
 --list_root './dataset_list/ucf_10per_paramshakti/' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-4 --epochs 200 --sched cosine --duration 8 --batch-size 24 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.3   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_60_sup_epoch_100_10per_ucf_bs24_gamma0.6_beta1_lr1e-4_temp0.5_wd0.3_drop_path0.3/' --hpe_to_token \
 --sup_thresh 200 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip --weight-decay 0.3 \
 --pretrained_path "./pretrainedCheckpoints/ckpt_epoch_60.pth" 
