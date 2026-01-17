import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

# --------------------
# Config
# --------------------
DATA_DIR = Path("data")
BATCH_SIZE = 16
NUM_EPOCHS = 20 #5 decrease epochs for quick testing
LR = 0.001
MODEL_PATH = "model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Transforms
# --------------------
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = train_transforms

# --------------------
# Datasets & Loaders
# --------------------
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------
# Model
# --------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# --------------------
# Training Loop
# --------------------
def run_epoch(model, loader, training=True):
    model.train() if training else model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.7)  # threshold 0.7
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total

# --------------------
# Train
# --------------------
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, training=True)
    val_loss, val_acc = run_epoch(model, val_loader, training=False)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

# --------------------
# Save Model
# --------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
