# train_emotion.py
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

DATASET_DIR = "emotions/DATASET"
MODEL_OUT = "models/emotion_model.pth"
os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=transform_train)
test_ds  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=transform_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 7)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Acc: {acc:.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "class_to_idx": train_ds.class_to_idx
}, MODEL_OUT)

print("Emotion model saved.")
