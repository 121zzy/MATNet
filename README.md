# MATNet: Multilevel Attention-Based Transformers for Change Detection in Remote Sensing Images
This repo is the PyTorch implementation of some works related to remote sensing tasks.
[`Wele Gedara Chaminda Bandara`](https://www.wgcban.com/), and [`Vishal M. Patel`](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)

:notebook_with_decorative_cover: **Accepted for publication at [`IGARSS-22`](https://www.igarss2022.org/default.php), Kuala Lumpur, Malaysia.**

## Environment setting
```
Python 3.6.13
pytorch 1.9.1
torchvision 0.10.1+cpu
```

Please see `requirements.txt` for all the other requirements.
You can create a virtual ``conda`` environment named ``ChangeFormer`` with the following cmd:

```
conda create --name MATNet --file requirements.txt
conda activate MATNet
```

###  Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.



You can download the processed LEVIR-CD and DSIFN-CD datasets by the DropBox through the following here:
- CDD:[`click here to download`](https://aistudio.baidu.com/datasetdetail/89523)
- LEVIR-CD-256: [`click here to download`](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip)
- DSIFN-CD-256: [`click here to download`](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip)
- SYSU-CD:[` Baidu Netdisk, code: tgrs`](https://pan.baidu.com/share/init?surl=rux9Zxjc8yGsga28CSD0kg)


## :speech_balloon: License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## :speech_balloon: Citation

If you use this code for your research, please cite our paper:

```

```

## :speech_balloon: References


