from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
from torchvision import transforms
from PIL import Image
from app.model import LungCNN
import io
import os
import base64

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))
app.mount(
    '/static',
    StaticFiles(directory=os.path.join(BASE_DIR, 'static')),
    name='static'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "model.pth")

model = LungCNN(num_classes=3)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    raise FileNotFoundError(f'Model file not found at {MODEL_PATH}')
model.to(device)
model.eval()

CLASS_NAMES = ['Benign', 'Malignant', 'Normal'] 

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/predict')
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = CLASS_NAMES[predicted.item()]

    buffered = io.BytesIO()
    image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_url = f'data:image/png;base64,{img_str}'

    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'prediction': pred_class,
            'filename': file.filename,
            'image_url': image_url
        }
    )
