import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import random

data = []
faces = 0.0
total = 0.0
faces_max=0
with open("wider_face_split\wider_face_train_bbx_gt.txt", 'r') as file:
    lines = file.readlines()
    idx = 0
    while idx < len(lines):
        image_path = "WIDER_train/WIDER_train/images/" + lines[idx].strip()
        idx += 1
        num_bounding_boxes = int(lines[idx].strip())
        idx += 1
        if num_bounding_boxes == 0:
            num_bounding_boxes = 1
        # Read bounding box information
        boxes_info = [lines[i].strip() for i in range(idx, idx + num_bounding_boxes)]
        # x y w h
        faces_max_round = 0
        for box_info in boxes_info:
            box_info = box_info.split(' ')
            if (0 != float(box_info[3])) and 0 != (float(box_info[2])):
                total += 1.0
                faces += 1.0
                faces_max_round += 1
            else:
                total += 1.0
        # padding 
        for i in range(1970-num_bounding_boxes):                  
            total += 1
        faces_max = max(faces_max, faces_max_round)
        idx += num_bounding_boxes
print(faces_max)
print(faces/total)