1) Download requirements
2) run server: uvicorn app.api.main:app
3) Use client part like this:
import requests
from PIL import Image
url = "http://your_host_url/detect/"
path = "path_to_image"
with open(path, "rb") as image:
    files = {"file": image}
    response = requests.post(url, files=files).json()["detections"]
image = Image.open(path)
for num, coord in enumerate(response):
    stamp_crop = image.crop(coord)
    stamp_crop.save(f"stamp_{num}.png")
image.close()
