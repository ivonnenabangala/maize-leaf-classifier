from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet50
from PIL import Image
import io

from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['blight', 'common_rust', 'gray_leaf_spot', 'healthy']


def load_model():
    model = resnet50(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4)
    )

    # model.load_state_dict(torch.load("resnet50_maize_model.pth", map_location=device))
    # model.load_state_dict(torch.load("resnet50_maize_model.pth", map_location=device, weights_only=False))
    state_dict = torch.load("resnet50_maize_model.pth", map_location=device)
    print("############")
    print("############")
    print("############")

    print(type(state_dict))   
    print("############")
    print("############")
    print("############")


    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
    

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())



@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    prediction = class_names[pred.item()]

    return {"prediction": prediction}