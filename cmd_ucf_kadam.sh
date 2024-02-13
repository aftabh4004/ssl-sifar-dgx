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
 --dataset_root '/scratch1/dataset/ucf101/videos/' \
 --output_dir './dataset_list/default' \
 --trainlist_path '/scratch1/dataset/ucf101/ucf101_train_split_1_videos.txt' \
 --testlist_path '/scratch1/dataset/ucf101/ucf101_val_split_1_videos.txt' \
 --percentage 10