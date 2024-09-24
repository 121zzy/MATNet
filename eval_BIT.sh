#!/usr/bin/env bash

gpus=1

data_name=DSIFN
net_G=DTCDSCN
split=test
vis_root=/media/dsk2/zhongyu.zhang/project/ChangeFormer/vis
project_name=DTCDSCN_DSIFN_lr0.01
checkpoints_root=/media/dsk2/zhongyu.zhang/project/ChangeFormer/checkpoints
checkpoint_name=best_ckpt.pt
img_size=256


python eval_cd.py --split ${split} --net_G ${net_G} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


