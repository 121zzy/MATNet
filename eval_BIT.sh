#!/usr/bin/env bash

gpus=1

data_name=DSIFN
net_G=MATNet
split=test
vis_root=/media/dsk2/zzy/project/MATNet/vis
project_name=CD_MATNet_DSIFN_b8_lr0.01_sgd_train_test_100_linear_ce
checkpoints_root=/media/dsk2/zzy/project/MATNet/checkpoints
checkpoint_name=best_ckpt.pt
img_size=256


python eval_cd.py --split ${split} --net_G ${net_G} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


