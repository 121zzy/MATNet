#!/usr/bin/env bash

gpus=0
checkpoint_root=/media/dsk2/zzy/project/ChangeFormer/checkpoints
vis_root=/media/dsk2/zzy/project/ChangeFormer/vis
data_name=LEVIR #LEVIR, DSIFN，CDD,SYSU

img_size=256
batch_size=8
lr=0.01
max_epochs=100
net_G=MATNet
lr_policy=linear

split=train
split_val=test
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}

CUDA_VISIBLE_DEVICES=0 python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
