import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

txt_file = "wider_face_split\wider_face_train_bbx_gt.txt"
with open(txt_file, 'r') as file:
            lines = file.readlines()
            idx = 0
            total_accuracy = 0
            times = 0
            while idx < len(lines):
                times += 1
                image_path = "WIDER_train/WIDER_train/images/" + lines[idx].strip()
                idx += 1
                num_bounding_boxes = int(lines[idx].strip())
                idx += 1
                image = Image.open(image_path).convert('RGB')
                width, height = image.size
                scale_x = 320/width
                scale_y = 320/height
                if num_bounding_boxes == 0:
                    num_bounding_boxes = 1
                # Read bounding box information
                boxes_info = [lines[i].strip() for i in range(idx, idx + num_bounding_boxes)]
                all = 0
                correct = 0
                for box_info in boxes_info:
                    box_info = box_info.split(' ')
                    min = 10
                    if (int((float(box_info[2])) * float(scale_x)) != min) and (int(float(box_info[3])* float(scale_y))!=min):
                        correct += 1
                    all += 1
                accuracy = correct / all
                total_accuracy += accuracy
                print('accuracy: ', accuracy)
                idx += num_bounding_boxes
            print('total accuracy: ', total_accuracy/times)