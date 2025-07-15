from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.model.model import YOLOStamp, yolo_pipeline
import torch

app = FastAPI()
model = YOLOStamp() # Инициализируем модель и загружаем в неё веса
model.load_state_dict(torch.load("app/api/weight.pt"))
@app.post("/detect/")
async def get_boxes(file: UploadFile = File(...)): # Единственный роут, принимает пост запрос с файлом
    image = Image.open(file.file)
    result = yolo_pipeline(model, "cpu", image)  # Можно указать необходимый тип устройства
    return JSONResponse(content={"detections": result.numpy().tolist()})
