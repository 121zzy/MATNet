import os
import cv2
import numpy as np

def process_images(directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for file_name in os.listdir(directory):
        if file_name.endswith(".png"):
            image_path = os.path.join(directory, file_name)
            image = cv2.imread(image_path)

            # Cut and save the image into patches
            index = 1
            for i in range(4):
                for j in range(4):
                    # Calculate the bounding box coordinates
                    y_min, y_max = i*256, (i+1)*256
                    x_min, x_max = j*256, (j+1)*256

                    # Cut the patch
                    patch = image[y_min:y_max, x_min:x_max]

                    # Save the patch
                    output_file_name = "{}_{}.png".format(file_name.split('.png')[0], index)
                    output_path = os.path.join(output_directory, output_file_name)
                    cv2.imwrite(output_path, patch)

                    index += 1

directories = [
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/train/A",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/train/A"
    },
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/train/B",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/train/B"
    },
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/train/label",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/train/label"
    },
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/test/A",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/test/A"
    },
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/test/B",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/test/B"
    },
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/test/label",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/test/label"
    },
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/val/A",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/val/A"
    },
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/val/B",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/val/B"
    },
    {
        "input": "/media/dsk2/zhongyu.zhang/project/LEVIR_CD/val/label",
        "output": "/media/dsk2/zhongyu.zhang/project/LEVIR-CD256/val/label"
    },
]

for directory in directories:
    process_images(directory["input"], directory["output"])