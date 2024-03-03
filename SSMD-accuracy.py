import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
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

def calculate_accuracy(ground_truth_boxes, predicted_boxes, iou_threshold=0.9):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for gt_box in ground_truth_boxes:
        found_match = False

        for pred_box in predicted_boxes:
            iou = calculate_iou(gt_box, pred_box)
            
            if iou >= iou_threshold:
                true_positives += 1
                found_match = True
                break

        if not found_match:
            false_negatives += 1

    false_positives = len(predicted_boxes) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = true_positives / len(ground_truth_boxes)

    return precision, recall, accuracy




output_size = 2000
# Load your trained ResNet-18 model
model = ssdlite320_mobilenet_v3_large(weights=None)
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

# Hypothetical example: Assume model outputs bounding box coordinates [x, y, width, height]
# Adjust this based on your actual output format
pred_scores = outputs[0]['scores'].detach().cpu().numpy()
bounding_box = outputs[0]['boxes'].detach().cpu().numpy()
labels = outputs[0]['labels'].detach().cpu().numpy()
boxes = bounding_box[pred_scores >= 0.4].astype(np.int32)
# Post-process the output
# You may need to convert model-specific output to a usable format for bounding boxes
# For example, if the model outputs bounding box deltas, you might need to apply them to anchor boxes


with open("wider_face_split\wider_face_val_bbx_gt.txt", 'r') as file:
    lines = file.readlines()
    idx = 0
    i = 0
    final_precision=0
    final_recall=0
    final_accuracy=0
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
        bounding_boxes_result = bounding_box[pred_scores >= 0.4].astype(np.int32)



        if num_bounding_boxes == 0:
            num_bounding_boxes = 1
        # Read bounding box information
        boxes_info = [lines[i].strip() for i in range(idx, idx + num_bounding_boxes)]
        # x y w h
        bounding_boxes_target = []
        for box_info in boxes_info:
            box_info = box_info.split(' ')
            x = 0
            y = 0
            if(float(box_info[2]) * float(scale_x) < 1):
                x = 1
            if(float(box_info[3]) * float(scale_y) < 1):
                y = 1
            bounding_boxes = [int(float(box_info[0]) * float(scale_x)),int(float(box_info[1]) * float(scale_y)),int((float(box_info[0])+float(box_info[2])) * float(scale_x))+x,int((float(box_info[1])+float(box_info[3])) * float(scale_y))+y]
            bounding_boxes_target.append(bounding_boxes)
        precision,recall, accuracy  = calculate_accuracy(bounding_boxes_target, bounding_boxes_result)
        idx += num_bounding_boxes
        print('precision = ', precision)
        print('recall = ', recall)
        print('accuracy = ', accuracy)
        final_precision +=precision
        final_recall += recall
        final_accuracy += accuracy
    print('final precision = ', final_precision/i)
    print('final recall = ', final_recall/i)
    print('final accuracy = ', final_accuracy/i)
