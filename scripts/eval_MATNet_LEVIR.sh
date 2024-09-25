#!/usr/bin/env bash

gpus=0

data_name=LEVIR
net_G=MATNet 
split=test
vis_root=/media/lidan/ssd2/MATNet/vis
project_name=CD_MATNet_LEVIR_b8_lr0.01_sgd_train_test_100_linear_ce
checkpoints_root=/media/lidan/ssd2/MATNet/checkpoints
checkpoint_name=best_ckpt.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

CUDA_VISIBLE_DEVICES=0 python eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


