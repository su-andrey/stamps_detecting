from model_pipeline import YoloStampPipeline
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
app = FastAPI()


@app.post("/detect/")
async def get_boxes(file: UploadFile = File(...)):
    pipeline = YoloStampPipeline.from_pretrained("stamps-labs/yolo-stamp", "cpu")
    print(file.file)
    image = Image.open(file.file)
    result = pipeline(image)
    return JSONResponse(content={"detections": result.numpy().tolist()})
