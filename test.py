import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import timm  # PyTorch Image Models (pretrained ViT)

# ==========================
# ⚡ Step 1: Define Vision Transformer Model
# ==========================
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super(VisionTransformer, self).__init__()
        
        # Load Pretrained ViT Model from timm library
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)

        # Modify Classification Head for CIFAR-10 (10 classes)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# ==========================
# ⚡ Step 2: Preprocessing the Input Image
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image for ViT input
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==========================
# ⚡ Step 3: Train Vision Transformer
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
for epoch in range(1):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/1], Loss: {loss.item():.4f}")

# ==========================
# ⚡ Step 4: Make a Prediction
# ==========================
def predict(image_path, model):
    from PIL import Image
    import torch.nn.functional as F

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print(f"Predicted Class: {predicted_class}")

# Example usage
predict("example_image.jpg", model)
