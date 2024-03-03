import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import numpy as np
output_size = 2000
# Load your trained ResNet-18 model
model = ssdlite320_mobilenet_v3_large(weights=None)
# Load the saved weights
model.load_state_dict(torch.load('trained_model.pth'))
# Set the model to evaluation mode
model.eval()

# Load and preprocess the input image
input_image_path = "WIDER_val/images/2--Demonstration/2_Demonstration_Demonstrators_2_470.jpg"
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
print(bounding_box)
print(labels)
# Post-process the output
# You may need to convert model-specific output to a usable format for bounding boxes
# For example, if the model outputs bounding box deltas, you might need to apply them to anchor boxes.

# Visualization
draw = ImageDraw.Draw(image)
for box, label in zip(boxes,labels):
    box_coords = [box[0] * image.width/320, box[1] * image.height/320,
                box[2] * image.width/320, box[3] * image.height/320]
    if box_coords != [0,0,0,0]:
        print(box_coords)
    draw.rectangle(box_coords, outline="red", width=2)


# Display or save the image with bounding box
image.show()
# image.save('path/to/save/output_image.jpg')