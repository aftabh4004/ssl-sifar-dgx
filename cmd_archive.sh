 CUDA_VISIBLE_DEVICES=0  python3 main.py --data_dir '/home/mt0/22CS60R54/sifar-pytorch/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-4 --epochs 20 --sched cosine --duration 8 --batch-size 8 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 5 --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output/' --hpe_to_token --initial_checkpoint '/home/mt0/22CS60R54/sifar-pytorch/sifar_base_patch4_window14_224_3x3-kinetics400_f8_pe_aug.pth'


  python -m torch.distributed.launch --nproc_per_node=6 main.py --data_dir [path-to-video] --use_pyav --dataset sth2stv2 \
 --opt adamw --lr 1e-4 --epochs 30 --sched cosine --duration 8 --batch-size 2 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 5 --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir [output_dir] --hpe_to_token --initial_checkpoint [path-to-pretrain] --eval --num_crops 3 --num_clips 3



CUDA_VISIBLE_DEVICES=0  python3  main.py --data_dir '/home/mt0/22CS60R54/sifar-pytorch/UCF-101' --use_pyav  --dataset 'ucf101' \
--opt adamw --lr 1e-4 --epochs 20 --sched cosine --duration 16 --batch-size 4 --super_img_rows 4 --disable_scaleup \
--mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 0 --no-amp --model sifar_base_patch4_window14_224_4x4 \
--output_dir './output4x4/' --hpe_to_token --initial_checkpoint '/home/mt0/22CS60R54/sifar-pytorch/sifar_base_patch4_window14_224_4x4-kinetics400_f16_pe_aug_v1.pth'



 CUDA_VISIBLE_DEVICES=0,1  python3 main.py --data_dir '/home/mt0/22CS60R54/sifar-pytorch/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 1e-5 --epochs 30 --sched cosine --duration 8 --batch-size 8 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 5 --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output3x3bs8lr10e-5/' --hpe_to_token --initial_checkpoint '/home/mt0/22CS60R54/sifar-pytorch/sifar_base_patch4_window14_224_3x3-kinetics400_f8_pe_aug.pth'


CUDA_VISIBLE_DEVICES=1  python3  main.py --data_dir '/home/mt0/22CS60R54/sifar-pytorch/UCF-101' --use_pyav  --dataset 'ucf101' \
--opt adamw --lr 5e-6 --epochs 30 --sched cosine --duration 16 --batch-size 4 --super_img_rows 4 --disable_scaleup \
--mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 0 --no-amp --model sifar_base_patch4_window14_224_4x4 \
--output_dir './output4x4bs8lr5e-6/' --hpe_to_token --initial_checkpoint '/home/mt0/22CS60R54/sifar-pytorch/sifar_base_patch4_window14_224_4x4-kinetics400_f16_pe_aug_v1.pth'



 CUDA_VISIBLE_DEVICES=0  python3 main.py --data_dir '/home/mt0/22CS60R54/ssl-sifar/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 5e-6 --epochs 30 --sched cosine --duration 8 --batch-size 2 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 5 --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output3x3_5_percenttemp/' --hpe_to_token --initial_checkpoint '/home/mt0/22CS60R54/sifar-pytorch/sifar_base_patch4_window14_224_3x3-kinetics400_f8_pe_aug.pth'

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2  main.py --data_dir '/home/mt0/22CS60R54/ssl-sifar/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 5e-6 --epochs 30 --sched cosine --duration 8 --batch-size 2 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 5 --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output3x3_5_percent/' --hpe_to_token --initial_checkpoint '/home/mt0/22CS60R54/sifar-pytorch/sifar_base_patch4_window14_224_3x3-kinetics400_f8_pe_aug.pth'

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2  main.py --data_dir '/home/mt0/22CS60R54/sifar-pytorch/UCF-101' --use_pyav  --dataset 'ucf101' \
--opt adamw --lr 1e-4 --epochs 20 --sched cosine --duration 16 --batch-size 2 --super_img_rows 4 --disable_scaleup \
--mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 0 --no-amp --model sifar_base_patch4_window14_224_4x4 \
--output_dir './output4x4/' --hpe_to_token --initial_checkpoint '/home/mt0/22CS60R54/sifar-pytorch/sifar_base_patch4_window14_224_4x4-kinetics400_f16_pe_aug_v1.pth'





 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  main.py --data_dir '/home/mt0/22CS60R54/ssl-sifar/UCF-101' --use_pyav --dataset 'ucf101' \
 --opt adamw --lr 5e-6 --epochs 30 --sched cosine --duration 8 --batch-size 2 --super_img_rows 3 --disable_scaleup \
 --mixup 0.8 --cutmix 1.0 --drop-path 0.1 --pretrained --warmup-epochs 5 --no-amp --model sifar_base_patch4_window14_224_3x3 \
 --output_dir './output3x3_5_percent_temp/' --hpe_to_token --initial_checkpoint '/home/mt0/22CS60R54/sifar-pytorch/sifar_base_patch4_window14_224_3x3-kinetics400_f8_pe_aug.pth'

