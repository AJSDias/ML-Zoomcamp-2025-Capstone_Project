import io
import torch
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms, models
from PIL import Image

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"
THRESHOLD = 0.7

# ----------------------------
# image transforms
# ----------------------------
transform = transforms.Compose([
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
# load model
# ----------------------------
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()


@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_t).squeeze(1)
        prob = torch.sigmoid(logits).item()

    label = "PNEUMONIA" if prob > THRESHOLD else "NORMAL"

    return {"label": label, "probability": prob}
