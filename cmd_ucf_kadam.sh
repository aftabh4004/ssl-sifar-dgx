python3  main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/run210pre_ucf_lr1e-5bs2mu4gamma0.6beta1temp0.5/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1


python3 -m torch.distributed.launch --standalone --nnodes=1 --nproc_per_node=4 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/run210pre_ucf_lr1e-5bs2mu4gamma0.6beta1temp0.5/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1


python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/test/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 10 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

 instance 1.46
 group 2.31

python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/OnlyUnsup_10per_ucf_bs4_mu7_gamma1.46_beta2.31_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 0 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 1.46 --beta 2.31


 python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/NoSup_10per_ucf_bs4_mu7_gamma1.46_beta2.31_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 0 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

  python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/contrastive_only_10per_ucf_bs4_mu7_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 0 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sup_epoch5_10per_ucf_bs4_mu7_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 5 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

 python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sup_epoch5_contrastive_only_10per_ucf_bs4_mu7_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 5 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

  python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sup_epoch5_contrastive_only_10per_ucf_bs2_mu7_gamma1_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 5 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 1 --beta 1

   python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sup_epoch150_10per_ucf_bs2_mu7_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 150 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

 python3 data_preparation_av.py \
 --dataset_root '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --output_dir './dataset_list/ucf_10per/' \
 --trainlist_path '/home/mt0/22CS60R54/datasets/ucf101/ucf101_train_split_1_videos.txt' \
 --testlist_path '/home/mt0/22CS60R54/datasets/ucf101/ucf101_val_split_1_videos.txt' \
 --percentage 10

 python3 main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/test/' --hpe_to_token \
 --sup_thresh 150 --num_workers 4 --mu 7 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1


# Reversible
 python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_sup_epoch25_ep150_10per_ucf_bs2_mu4_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained



python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 50 --sched cosine --duration 8 --batch-size 50 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_with_ckpt_only_sup_epoch_50_10per_ucf_bs50_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained


 python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 50 --sched cosine --duration 8 --batch-size 50 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_scratch_only_sup_epoch_50_10per_ucf_bs50_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 

 python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 100 --sched cosine --duration 8 --batch-size 50 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_scratch_only_sup_epoch_50_10per_ucf_bs50_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 


//gnode 3
 python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 50 --sched cosine --duration 8 --batch-size 50 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_ckpt_norm_only_sup_epoch_50_10per_ucf_bs50_gamma0.6_beta1_lr2e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained

//gnode4
 python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 50 --sched cosine --duration 8 --batch-size 48 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_without_ckpt_only_sup_epoch_50_10per_ucf_bs48_gamma0.6_beta1_lr2e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

 //gnode2

 python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 50 --sched cosine --duration 8 --batch-size 48 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_scratch_only_sup_epoch_50_10per_ucf_bs48_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 50 --sched cosine --duration 8 --batch-size 48 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/test/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained
 
 
//For revswin scratch noflip epoch 100
//gnode2 screen revswin
python3 main.py  --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/'  --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per'  --use_pyav --dataset 'ucf101'  --opt adamw --lr 2e-6 --epochs 100 --sched cosine --duration 8 --batch-size 46  --super_img_rows 3 --disable_scaleup  --mixup 0.8 --cutmix 1.0 --drop-path 0.1  --model sifar_base_patch4_window12_192_3x3  --output_dir './output/revswin_scratch_only_noflip_sup_epoch_100_10per_ucf_bs46_gamma0.6_beta1_lr2e-6_temp0.5/' --hpe_to_token  --sup_thresh 100 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5  --gamma 0.6 --beta 1


//For revswin ckpt noflip epoch 100 wd 0.3
//gnode4 screen revswin_scratch
python3 main.py  --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/'  --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per'  --use_pyav --dataset 'ucf101'  --opt adamw --lr 1e-3 --epochs 100 --sched cosine --duration 8 --batch-size 48  --super_img_rows 3 --disable_scaleup  --mixup 0.8 --cutmix 1.0 --drop-path 0.1    --model sifar_base_patch4_window12_192_3x3  --output_dir './output/revswin_ckpt_norm_only_noflip_sup_epoch_100_10per_ucf_bs48_gamma0.6_beta1_lr1e-3_temp0.5_wd_0.3/' --hpe_to_token  --sup_thresh 100 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5  --gamma 0.6 --beta 1 --pretrained --weight-decay 0.3




python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 50 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/swin_pretrained_sup_epoch_50_10per_ucf_bs2_gamma0.6_beta1_lr1e-5_temp0.5/' --hpe_to_token \
 --sup_thresh 50 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained

// gnode 4 
// screen revswin_scratch
python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 100 --sched cosine --duration 8 --batch-size 48 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_sup_epoch_100_10per_ucf_bs48_gamma0.6_beta1_lr2e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 100 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained \
 --pretrained_path "/home/mt0/22CS60R54/ssl-sifar-dgx/pretrainedCheckpoints/checkpoint_0.pth"



// gnode 2
// screen revswin
// no flip
python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 5e-4 --epochs 200 --sched cosine --duration 8 --batch-size 48 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_scratch_no_flip_sup_epoch_200_10per_ucf_bs48_gamma0.6_beta1_lr5e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 200 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --no_flip


// gnode 4
// screen revswin_scratch
// with flip
python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 5e-4 --epochs 200 --sched cosine --duration 8 --batch-size 48 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_scratch_with_flip_sup_epoch_200_10per_ucf_bs48_gamma0.6_beta1_lr5e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 200 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1



// gnode 4
// screen check
// no flip
python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 200 --sched cosine --duration 8 --batch-size 48 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_60_sup_epoch_200_10per_ucf_bs48_gamma0.6_beta1_lr2e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 200 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "/home/mt0/22CS60R54/ssl-sifar-dgx/pretrainedCheckpoints/ckpt_epoch_60.pth" 


// gnode 2
// screenrevswin
// with flip
python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 200 --sched cosine --duration 8 --batch-size 46 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_60_with_flip_sup_epoch_200_10per_ucf_bs46_gamma0.6_beta1_lr2e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 200 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained   \
 --pretrained_path "/home/mt0/22CS60R54/ssl-sifar-dgx/pretrainedCheckpoints/ckpt_epoch_60.pth" 



python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/ucf101/videos/' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/ucf_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2e-4 --epochs 200 --sched cosine --duration 8 --batch-size 5 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/test/' --hpe_to_token \
 --sup_thresh 0 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  --no_flip \
 --pretrained_path "/home/mt0/22CS60R54/ssl-sifar-dgx/pretrainedCheckpoints/ckpt_epoch_60.pth" 


salloc --gres=gpu:2 --time 23:00:00 --partition=gpupart_q4000 --nodes=1  srun --pty /bin/bash


// g4
// screen -r 960662.node_req2
python3 main.py \
 --data_dir '/home/mt0/22CS60R54/datasets/hmdb51' \
 --list_root '/home/mt0/22CS60R54/ssl-sifar-dgx/dataset_list/hmdb51_40per_SVformer' \
 --use_pyav --dataset 'hmdb51' \
 --opt adamw --lr 2e-4 --epochs 200 --sched cosine --duration 8 --batch-size 46 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1   \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/revswin_pretrained_from_ckpt_178_with_flip_sup_epoch_200_40per_hmdb_bs46_gamma0.6_beta1_lr2e-4_temp0.5/' --hpe_to_token \
 --sup_thresh 200 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --pretrained  \
 --pretrained_path "/home/mt0/22CS60R54/ssl-sifar-dgx/pretrainedCheckpoints/ckpt_epoch_178.pth" 