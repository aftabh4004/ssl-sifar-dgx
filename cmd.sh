CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2.5e-6 --epochs 150 --sched cosine --duration 8 --batch-size 2 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output/test' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 224


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0  --master_port=28700 main.py --data_dir '/scratch/datasets/UCF-101/Videos' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-4 --epochs 30 --sched cosine --duration 8 --batch-size 2 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output/' --hpe_to_token


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 5e-6 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr5e-6/testtest' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192


CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 2.5e-6 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr5e-6/testtest' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-5/' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-5temp0.3/' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.3

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-3 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-3temp0.3/' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.3


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-5temp0.5icYes/' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 \
 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/sifar_base_patch4_window14_224_3x3-kinetics400_f8_pe_aug.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-5-6-7temp0.3/' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.3 


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/test' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.8 





CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-5-6-7temp0.8minist2stv2' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.8 


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-3 --epochs 150 --sched step --decay-epochs 5 --decay-rate 0.1 --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lrsteptemp0.5minist2stv2' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 --weight-decay 0.0

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-3 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lrcosintemp0.5minist2stv2' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 --weight-decay 0.0


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 150 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-5cosintemp0.5minist2stv2' --hpe_to_token --sup_thresh 25 --num_workers 4 --mu 2 --input_size 192 --temperature 0.8 



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 5e-3 --epochs 400 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr5e-3cosintemp0.5minist2stv2' --hpe_to_token --sup_thresh 50 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 

lr 0.001, bs 8 per gpu
--weight-decay 0.0
50 epoch super
400 epoch

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 5e-3 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-5cosinWUep35ep400Sep0temp0.5minist2stv2' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/192bs4se50ep400lr1e-5cosintemp0.5minist2stv2/model_best.pth \
 --warmup-epochs 35
 


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/192bs4lr1e-5cosinWUep35ep400Sep0temp0.5minist2stv2exp2' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/192bs4se50ep400lr1e-5cosintemp0.5minist2stv2/model_best.pth \
 --warmup-epochs 35
 

 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-3 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonlylr1e-3WUep35Ep350temp0.5' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5 \
 --warmup-epochs 35 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly50lr1e-5temp0.5ep50schedCos/checkpoint.pth 
 
 


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 50 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/suponly50lr1e-5temp0.5ep50schedCos' --hpe_to_token --sup_thresh 50 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 

/home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly50lr1e-5temp0.5ep50schedCos/model_best.pth


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 50 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/test' --hpe_to_token --sup_thresh 50 --num_workers 4 --mu 2 --input_size 192 --temperature 0.5 \
 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly50lr1e-5temp0.5ep50schedCos/checkpoint.pth \
 --eval


 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-3 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonlylr1e-3WUep35Ep350temp0.5' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5 \
 --warmup-epochs 35 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly50lr1e-5temp0.5ep50schedCos/checkpoint.pth 
 

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 3.5e-5 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonlylr1e-3WUep35Ep350temp0.3' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.3 \
 --warmup-epochs 0 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly50lr1e-5temp0.5ep50schedCos/checkpoint.pth 
 


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 5e-5 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonlylr5e-5WUep35Ep350temp0.5' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5 \
 --warmup-epochs 0 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly50lr1e-5temp0.5ep50schedCos/checkpoint.pth 
 


gamma 5, lr 1e-2

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-3 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonlylr1e-3temp0.5gamma5' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5  --warmup-epochs 0 \
 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly50lr1e-5temp0.5ep50schedCos/checkpoint.pth \
 --gamma 5.0 


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 3e-5 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonlylr3e-5temp0.5gamma2' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5  --warmup-epochs 0 \
 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly50lr1e-5temp0.5ep50schedCos/checkpoint.pth \
 --gamma 2.0 


 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 50 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/suponly20percent50lr1e-5temp0.5ep50schedCos' --hpe_to_token --sup_thresh 50 --num_workers 4 \
 --mu 2 --input_size 192 --temperature 0.5 


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 3e-5 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonly20perlr3e-5temp0.5gamma2beta3' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5  --warmup-epochs 0 \
 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly20percent50lr1e-5temp0.5ep50schedCos/checkpoint.pth \
 --gamma 2.0 --beta 3.0 

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 50 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/suponly10percent50lr1e-5temp0.5ep50schedCos' --hpe_to_token --sup_thresh 50 --num_workers 4 \
 --mu 2 --input_size 192 --temperature 0.5 

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 3e-5 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonlywarmup35lr3e-5from1e-7' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5  --warmup-epochs 0 \
 --initial_checkpoint /home/prithwish/aftab/workspace/ssl-sifar-dgx/output/suponly20percent50lr1e-5temp0.5ep50schedCos/checkpoint.pth \
 --gamma 2.0 --beta 0.5


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 350 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/sslonlywarmup35lr3e-5from1e-7' --hpe_to_token --sup_thresh 0 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5  --warmup-epochs 0 \
 --gamma 2.0 --beta 0.5


 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 400 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/lr1e-5cosinebs4mu2ep400sup50temp0.5gamma1beta1_5percent' --hpe_to_token --sup_thresh 50 --num_workers 4 --mu 2 \
 --input_size 192 --temperature 0.5  --warmup-epochs 0 \
 --gamma 1 --beta 1

  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 400 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/Run2lr1e-5cosinebs4mu1ep400sup50temp0.5gamma1beta1_5percent' --hpe_to_token --sup_thresh 50 --num_workers 4 --mu 1 \
 --input_size 192 --temperature 0.5  --warmup-epochs 0 \
 --gamma 1 --beta 1

#  psedo threshold reduce and see if geeting any   

group 6, instance 2


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py  --dataset 'mini_st2stv2' \
 --opt adamw --lr 1e-5 --epochs 400 --sched cosine --duration 8 --batch-size 4 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --model sifar_base_patch4_window12_192_3x3 \
 --output_dir './output/lr1e-5cosinebs4mu1ep400sup50temp0.5gamma2beta6_5percent' --hpe_to_token --sup_thresh 50 --num_workers 4 --mu 1 \
 --input_size 192 --temperature 0.5  --warmup-epochs 0 \
 --gamma 2 --beta 6
