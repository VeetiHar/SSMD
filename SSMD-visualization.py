import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import os
import numpy as np
output_size = 2
# Load your trained ResNet-18 model
model = ssdlite320_mobilenet_v3_large(num_classes=output_size)
# Load the saved weights
model.load_state_dict(torch.load('trained_model.pth'))
# Set the model to evaluation mode
model.eval()

# Load and preprocess the input image
input_image_path0 = "WIDER_val/images/13--Interview/13_Interview_Interview_On_Location_13_334.jpg"
input_image_path1 = "WIDER_val/images/37--Soccer/37_Soccer_Soccer_37_618.jpg"
input_image_path2 = "images/crosswalk.jpg"
input_image_path3 = "images/group2.jpg"
input_image_path4 = "images/Jackie-Chan-Shutterstock.jpg"
image_array = [input_image_path0, input_image_path1, input_image_path2, input_image_path3, input_image_path4]
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    # Add any other preprocessing steps used during training
])
i = 0
for input_image_path in image_array:
    image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)

    # Hypothetical example: Assume model outputs bounding box coordinates [x, y, width, height]
    # Adjust this based on your actual output format
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    bounding_box = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].detach().cpu().numpy()
    boxes = bounding_box[pred_scores >= 0.2].astype(np.int32)

    # Post-process the output
    # You may need to convert model-specific output to a usable format for bounding boxes
    # For example, if the model outputs bounding box deltas, you might need to apply them to anchor boxes.
    draw = ImageDraw.Draw(image)
    if i == 0:
        box_coords0 = [211, 267, 211+87, 267+116]
        box_coords1 = [513, 255, 513+78 ,255+118]
        box_coords2 = [746, 260, 746+76 ,260+109]
        draw.rectangle(box_coords0, outline="green", width=2)
        draw.rectangle(box_coords1, outline="green", width=2)
        draw.rectangle(box_coords2, outline="green", width=2)
    if i == 1:
        box_coords0 = [10, 304, 10+74, 304+90]
        box_coords1 = [242, 322, 242+74, 322+94]
        box_coords2 = [488, 334, 488+84, 334+114]
        box_coords3 = [822, 360, 822+72, 360+116]
        box_coords4 = [662, 334, 662+68 ,334+98]
        box_coords5 = [460, 290, 460+66, 290+86]
        draw.rectangle(box_coords0, outline="green", width=2)
        draw.rectangle(box_coords1, outline="green", width=2)
        draw.rectangle(box_coords2, outline="green", width=2)
        draw.rectangle(box_coords3, outline="green", width=2)
        draw.rectangle(box_coords4, outline="green", width=2)
        draw.rectangle(box_coords5, outline="green", width=2)
    # Visualization
    for box, label in zip(boxes,labels):
        if label == 1:
            box_coords = [box[0] * image.width/300, box[1] * image.height/300,
                        box[2] * image.width/300, box[3] * image.height/300]
            print(box_coords)
            draw.rectangle(box_coords, outline="red", width=2)
    i += 1

    # Display or save the image with bounding box
    #image.show()
    os.makedirs('recognized', exist_ok=True)
    image.save('recognized/rec_output_image'+str(i)+'.jpg')