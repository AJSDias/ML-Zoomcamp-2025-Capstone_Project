import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
BATCH_SIZE = 16
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.7

# ----------------------------
# transforms
# ----------------------------
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# dataset + loader
# ----------------------------
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=test_transforms)
print("Class mapping:", test_dataset.class_to_idx)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ----------------------------
# load model
# ----------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

# ----------------------------
# evaluation
# ----------------------------
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs)

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.tolist())

# compute metrics
tp = fp = tn = fn = 0

for p, l in zip(all_probs, all_labels):
    pred = 1 if p > THRESHOLD else 0

    if pred == 1 and l == 1:
        tp += 1
    elif pred == 1 and l == 0:
        fp += 1
    elif pred == 0 and l == 0:
        tn += 1
    elif pred == 0 and l == 1:
        fn += 1

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)

print("Confusion matrix:")
print(f"TP: {tp}, FP: {fp}")
print(f"FN: {fn}, TN: {tn}")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
