CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/10_per_bs4lr1e-5/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/10_per_bs4lr1e-5/' --hpe_to_token \
 --sup_thresh 0 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 3 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/10_per_bs3lr1e-5mu3/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 3 --input_size 192 --temperature 0.5

 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 6 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/10_per_bs6lr1e-5mu1/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 1 --input_size 192 --temperature 0.5

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 6 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/10_per_bs6lr1e-5mu1gamma2/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 1 --input_size 192 --temperature 0.5 \
 --gamma 2 
 

   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/10_per_bs4lr1e-5mu2gamma2beta2/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 \
 --gamma 2 --beta 2

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/10_per_bs2lr1e-5mu4gamma1beta1/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 1 --beta 1

 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/10_per_bs2lr1e-5mu4gamma0.6beta1/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/1_per_bs2lr1e-5mu4gamma1beta1/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 1 --beta 1

 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/1_per_bs2lr1e-5mu4gamma0.6beta1/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/1_per_bs4lr1e-5mu2gamma2beta1/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 \
 --gamma 2 --beta 1

   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/1_per_bs4lr1e-5mu2gamma2beta2/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 \
 --gamma 2 --beta 2

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 6 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/1_per_bs6lr1e-5mu1gamma1beta1/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 1 --input_size 192 --temperature 0.5 \
 --gamma 1 --beta 1



  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 35 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/temp_10_per_bs2lr1e-5mu4gamma0.6beta1/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1


   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py \
 --data_dir '/home/prithwish/aftab/dataset/UCF-101' \
 --list_root '/home/prithwish/aftab/workspace/ssl-sifar-dgx/dataset_list/ucf10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/test/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1 --no-amp


python3  main.py \
 --data_dir '/scratch1/dataset/ucf101/videos' \
 --list_root '/home/omprakash/aftab/ssl-sifar-dgx/dataset_list/kadamini/ucf101_10per' \
 --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 2 \
 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  \
 --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/run2recreateres_10pre_ucf_lr1e-5bs2mu4gamma0.6beta1temp0.5/' --hpe_to_token \
 --sup_thresh 25 --num_workers 4 --mu 4 --input_size 192 --temperature 0.5 \
 --gamma 0.6 --beta 1


 