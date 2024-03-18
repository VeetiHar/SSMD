import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def calculate_accuracy(ground_truth_boxes, predicted_boxes, pred_scores, hits , scores, iou_threshold=0.95):
    for gt_box in ground_truth_boxes:
        found_match = False

        for pred_box, score in zip(predicted_boxes,pred_scores):
            iou = calculate_iou(gt_box, pred_box)
            
            if iou >= iou_threshold:
                found_match = True
                scores.append(score)
                hits.append(1)
                break

        if not found_match:
            scores.append(score)
            hits.append(0)

    return hits, scores


output_size = 91
# Load your trained ResNet-18 model
model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=output_size)
# Load the saved weights
model.load_state_dict(torch.load('trained_model.pth'))
# Set the model to evaluation mode
model.eval()

# Load and preprocess the input image
input_image_path = 'WIDER_val/images/0--Parade/0_Parade_marchingband_1_353.jpg'
image = Image.open(input_image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    # Add any other preprocessing steps used during training
])
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Forward pass
with torch.no_grad():
    outputs = model(input_tensor)

hits = []
scores = []

with open("wider_face_split\wider_face_val_bbx_gt.txt", 'r') as file:
    lines = file.readlines()
    idx = 0
    i = 0
    while idx < len(lines):
        i += 1
        image_path = "WIDER_val/images/" + lines[idx].strip()
        idx += 1
        num_bounding_boxes = int(lines[idx].strip())
        idx += 1
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        width, height = image.size
        scale_x = 320/width
        scale_y = 320/height
        
        with torch.no_grad():
            outputs = model(input_tensor)

        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        bounding_box = outputs[0]['boxes'].detach().cpu().numpy()
        labels = outputs[0]['labels'].detach().cpu().numpy()

        if num_bounding_boxes == 0:
            num_bounding_boxes = 1
        # Read bounding box information
        boxes_info = [lines[i].strip() for i in range(idx, idx + num_bounding_boxes)]
        # x y w h
        bounding_boxes_target = []
        for box_info, label in zip(boxes_info, labels):
            box_info = box_info.split(' ')
            x = 0
            y = 0
            if(float(box_info[2]) * float(scale_x) < 1):
                x = 1
            if(float(box_info[3]) * float(scale_y) < 1):
                y = 1
            if label == 1:
                bounding_boxes = [int(float(box_info[0]) * float(scale_x)),int(float(box_info[1]) * float(scale_y)),int((float(box_info[0])+float(box_info[2])) * float(scale_x)),int((float(box_info[1])+float(box_info[3])) * float(scale_y))]
                bounding_boxes_target.append(bounding_boxes)
        hits, scores  = calculate_accuracy(bounding_boxes_target, bounding_box, pred_scores, hits, scores)
        idx += num_bounding_boxes
# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(hits, scores)
print('hits = ',hits)
print('scores = ', scores)
# Compute average precision (AP)
ap = auc(recall, precision)
print(ap)