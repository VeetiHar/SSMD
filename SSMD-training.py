import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from custom_dataset import CustomDataset
import torch.optim as optim
import torch.nn.functional as F

#print(torch.cuda.is_available())
#print(torch.version.cuda)
#cuda_id = torch.cuda.current_device()
#print(torch.cuda.current_device())
       
#print(torch.cuda.get_device_name(cuda_id))

output_size = 2
pretrained = False

# Set up the paths and parameters
data_folder = "WIDER_train\WIDER_train\images"
txt_file = "wider_face_split\wider_face_train_bbx_gt.txt"

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# Create a custom dataset
custom_dataset = CustomDataset(txt_file=txt_file, transform=transform)

# Create a data loader
batch_size = 4
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
# Set up the model
model = ssdlite320_mobilenet_v3_large(num_classes=output_size,positive_fraction=0.001)

# Set up optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

# Training loop
num_epochs = 10  # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        list_of_dicts = [
        {key: value[i] for key, value in targets.items()}
        for i in range(len(list(targets.values())[0]))]
        list_of_dicts_cuda = [
            {key: value.clone().detach().to('cuda') for key, value in d.items()}
            for d in list_of_dicts
        ]       

        optimizer.zero_grad()
        # Forward pass
        # Extract predicted bounding boxes from the model output
        outputs = model(inputs, list_of_dicts_cuda)
        # Compute the loss
        loss = outputs['bbox_regression'] + outputs['classification']
        total_loss += loss
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    mean_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_loss}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
