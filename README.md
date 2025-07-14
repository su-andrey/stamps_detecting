# Stamp Detection service
API for automated stamp detection in documents
- **YOLO-based detection**
- **Multi-format support** - detect stamps from .png, .jpg, .jpeg, .tiff 
## Installation and launch
1. Clone the repository:
```bash
git clone https://github.com/su-andrey/stamps_detecting
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run server
```bash
uvicorn app.api.main:app
```
## API usage
Example of basic request for stamps detection
```python
import requests
from PIL import Image
url = "http://your_host:port/detect/"
path = "path_to_target_image"
with open(path, "rb") as image:
    files = {"file": image}
    response = requests.post(url, files=files).json()["detections"]
image = Image.open(path)
for num, coord in enumerate(response):
    stamp_crop = image.crop(coord)
    stamp_crop.save(f"stamp_{num}.png")
image.close()
```
## API response format
```json
{'detections': [[600.1234567890123, 380.1234567891234, 750.0019091090875, 520.95090790875], [55.55533399911999, 333.6250005175780, 190.5443878778828, 475.0750750750750]]}
```
The value corresponding to the "detections" key contains a list of lists of boxes coordinates in the format [x1, y1, x2, y2] - coordinates of the upper-left and lower-right corners
