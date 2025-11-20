# train_age.py
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BASE_DIR = "ages/utkface_aligned_cropped"
MODEL_OUT = "models/age_model.pth"
os.makedirs("models", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Auto UTKFace Dataset
# ----------------------------
class UTKFaceDataset(Dataset):
    def __init__(self, base_dir):
        self.files = []

        # Scan ALL subfolders (UTKFace + crop_part1)
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(".jpg"):
                    self.files.append(os.path.join(root, f))

        print("Total images found:", len(self.files))

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")

        # Extract age from filename (first part BEFORE _ )
        filename = os.path.basename(path)
        age = int(filename.split("_")[0])

        img = self.transform(img)
        return img, torch.tensor(age, dtype=torch.float32)

# ----------------------------
# Load Dataset
# ----------------------------
dataset = UTKFaceDataset(BASE_DIR)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----------------------------
# Create Model (Regression)
# ----------------------------
model = models.resnet34(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# Training Loop
# ----------------------------
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save({
    "model_state_dict": model.state_dict(),
}, MODEL_OUT)

print("Age model saved successfully!")
