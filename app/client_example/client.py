import requests
from PIL import Image
url = "http://localhost:8000/detect/"
path = "images/img1.png"
with open(path, "rb") as image:
    files = {"file": image}
    response = requests.post(url, files=files).json()["detections"]
image = Image.open(path)
for num, coord in enumerate(response):
    stamp_crop = image.crop(coord)
    stamp_crop.save(f"stamp_{num}.png")
image.close()