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
 --opt adamw --lr 2e-4 --epochs 200 --sched cosine --duration 8 --batch-size 46 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_178_sup_epoch_100_10per_ucf_bs46_gamma0.6_beta1_lr2e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 200 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "./ckpt_epoch_178.pth" 


// gpu019 login04
// screen gpu019
// no flip
python3 main.py \
 --data_dir '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' \
 --list_root './dataset_list/ucf_10per_paramshakti/' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 200 --sched cosine --duration 8 --batch-size 46 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_298_sup_epoch_200_10per_ucf_bs46_gamma0.6_beta1_lr2e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 200 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "./pretrainedcheckpoints/ckpt_epoch_298.pth" 


// gpu006 login04
// screen gpu006
// no flip
python3 main.py \
 --data_dir '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' \
 --list_root './dataset_list/ucf_10per_paramshakti/' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 3.5e-5 --epochs 200 --sched cosine --duration 8 --batch-size 7 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_298_sup_epoch_25_epoch150_10per_ucf_bs7_mu_4_gamma0.6_beta1_lr3.5e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "./pretrainedcheckpoints/ckpt_epoch_298.pth" 


// gpu012 login04
// screen gpu012
// no flip
python3 main.py \
 --data_dir '/scratch/20cs91r11/dataset/hmdb51/videos' \
 --list_root './dataset_list/hmdb51_40per_SVformer/' \
 --use_pyav --dataset 'hmdb51' \
 --opt adamw --lr 2.25e-5 --epochs 200 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_298_sup_epoch_25_epoch150_10per_hmdb51_bs2_mu_7_gamma0.6_beta1_lr2.25e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "./pretrainedcheckpoints/ckpt_epoch_298.pth" 



CUDA_VISIBLE_DEVICES=0 python3 main.py \
 --data_dir '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' \
 --list_root './dataset_list/ucf_10per_paramshakti/' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 200 --sched cosine --duration 8 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_298_sup_epoch_25_epoch150_10per_ucf_bs7_mu_4_gamma0.6_beta1_lr3.5e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 0 --num_workers 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "./pretrainedcheckpoints/ckpt_epoch_298.pth" \
 --batch-size 3 --mu 4









// gpu012 login04
// screen gpu012
python3 main.py \
 --data_dir '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' \
 --list_root './dataset_list/ucf101_10per_SVformer/' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 200 --sched cosine --duration 8 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_298_sup_epoch_50_epoch200_10per_ucf_bs2_mu_11_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "./pretrainedcheckpoints/ckpt_epoch_298.pth" \
 --batch-size 2 --mu 11


// gpu006 login04
// screen gpu006
python3 main.py \
 --data_dir '/home/20cs91r11/scratch_20cs91r11/dataset/ucf101/videos/' \
 --list_root './dataset_list/ucf101_10per_SVformer/' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-5 --epochs 200 --sched cosine --duration 8 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_298_sup_epoch_50_epoch200_10per_ucf_bs5_mu_7_gamma0.6_beta1_lr2e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "./pretrainedcheckpoints/ckpt_epoch_298.pth" \
 --batch-size 5 --mu 7


// gpu019 login04
// screen gpu019
// no flip
python3 main.py \
 --data_dir '/scratch/20cs91r11/aftab/dataset/hmdb51/' \
 --list_root './dataset_list/hmdb51_40per_SVformer/' \
 --use_pyav --dataset 'hmdb51' \
 --opt adamw --lr 1e-5 --epochs 200 --sched cosine --duration 8 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_298_sup_epoch_50_epoch150_10per_hmdb51_bs2_mu_11_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "./pretrainedcheckpoints/ckpt_epoch_298.pth" \
 --batch-size 2 --mu 11
