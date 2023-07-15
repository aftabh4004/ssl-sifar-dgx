CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4  main.py --data_dir '/home/prithwish/aftab/dataset/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-4 --epochs 150 --sched cosine --duration 8 --batch-size 2 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output/' --hpe_to_token --sup_thresh 25


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0  --master_port=28700 main.py --data_dir '/scratch/datasets/UCF-101/Videos' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-4 --epochs 30 --sched cosine --duration 8 --batch-size 2 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained  --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output/' --hpe_to_token