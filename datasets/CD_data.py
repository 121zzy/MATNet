import os
from PIL import Image
import numpy as np
from torch.utils import data
from datasets.data_utils import CDDataAugmentation

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"
IGNORE = 255
label_suffix='.jpg'  # 根据新的结构，标签文件后缀应为.jpg

# 以下的函数和类定义保持不变

# 更新get_img_post_path、get_img_path和get_label_path函数
def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name)

def load_img_name_list(dataset_path):
    """
    从指定文件中加载图像文件名列表。

    参数：
    dataset_path（str）：包含图像文件名列表的文件路径。

    返回：
    img_name_list（list of str）：图像文件名的列表。
    """
    # 使用numpy的loadtxt函数加载文本文件，dtype设置为str以确保读取为字符串
    img_name_list = np.loadtxt(dataset_path, dtype=str)

    # 如果加载的数据维度为2，则返回第一列，否则直接返回加载的数据
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

# 更新数据集目录结构的加载方式
class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, name)
        B_path = get_img_post_path(self.root_dir, name)

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img, img_B], _ = self.augm.transform([img, img_B], [], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        return self.A_size

# CDDataset类保持不变
class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        label = np.array(Image.open(L_path), dtype=np.uint8)
        # if you are getting error because of dim mismatch ad [:,:,0] at the end

        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255

        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        # print(label.max())

        return {'name': name, 'A': img, 'B': img_B, 'L': label}