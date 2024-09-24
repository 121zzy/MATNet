import os
import cv2
from shutil import copy
import tarfile
import zipfile

# 解压缩DSFIN数据集
zip_file_path = '/home/zhongyu.zhang/project/DSIFN1/DSIFN.tar.gz'
extracted_folder_path = '/home/zhongyu.zhang/project/DSIFN1'

with tarfile.open(zip_file_path, 'r:gz') as tar:
    tar.extractall(path=extracted_folder_path)
train_zip_file_path = os.path.join(extracted_folder_path, 'train.zip')
train_extracted_folder_path = os.path.join(extracted_folder_path, 'train')
with zipfile.ZipFile(train_zip_file_path, 'r') as train_zip:
    train_zip.extractall(path=train_extracted_folder_path)
# 创建存储小图块的目录结构
output_root = 'DSIFN_256'
sets = ['train', 'val', 'test']
categories = ['A', 'B', 'label']

for set_name in sets:
    for category in categories:
        os.makedirs(os.path.join(output_root, set_name, category), exist_ok=True)

# 遍历训练、验证和测试集
for set_name in sets:
    # 处理A图像
    imgs_path_A = os.path.join(extracted_folder_path, set_name, 't1')
    output_path_A = os.path.join(output_root, set_name, 'A')

    for img_file_name in os.listdir(imgs_path_A):
        img_path = os.path.join(imgs_path_A, img_file_name)
        img = cv2.imread(img_path)

        c = 1
        for j in range(2):
            for k in range(2):
                patch = img[j * 256:(j + 1) * 256, k * 256:(k + 1) * 256, :]
                patch_name = f'{img_file_name[:-4]}_{c}.png'
                patch_output_path = os.path.join(output_path_A, patch_name)
                cv2.imwrite(patch_output_path, patch)
                c += 1

    # 处理B图像
    imgs_path_B = os.path.join(extracted_folder_path, set_name, 't2')
    output_path_B = os.path.join(output_root, set_name, 'B')

    for img_file_name in os.listdir(imgs_path_B):
        img_path = os.path.join(imgs_path_B, img_file_name)
        img = cv2.imread(img_path)

        c = 1
        for j in range(2):
            for k in range(2):
                patch = img[j * 256:(j + 1) * 256, k * 256:(k + 1) * 256, :]
                patch_name = f'{img_file_name[:-4]}_{c}.png'
                patch_output_path = os.path.join(output_path_B, patch_name)
                cv2.imwrite(patch_output_path, patch)
                c += 1

    # 处理标签图像
    labels_path = os.path.join(extracted_folder_path, set_name, 'mask')
    output_label_path = os.path.join(output_root, set_name, 'label')

    for label_file_name in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_file_name)
        label = cv2.imread(label_path)

        if label_file_name.endswith('.tif'):
            # 如果标签是.tif格式，将其保存为.png格式
            label_name = f'{label_file_name[:-4]}.png'
            label_output_path = os.path.join(output_label_path, label_name)
            cv2.imwrite(label_output_path, label)
        else:
            # 如果标签是.png格式，直接复制到目标文件夹
            label_output_path = os.path.join(output_label_path, label_file_name)
            copy(label_path, label_output_path)

        c = 1
        for j in range(2):
            for k in range(2):
                patch = label[j * 256:(j + 1) * 256, k * 256:(k + 1) * 256, :]
                patch_name = f'{label_file_name[:-4]}_{c}.png'
                patch_output_path = os.path.join(output_label_path, patch_name)
                cv2.imwrite(patch_output_path, patch)
                c += 1
