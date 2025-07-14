from model import YoloStampPipeline
from PIL import Image

if __name__ == "__main__":
    pipeline = YoloStampPipeline.from_pretrained("stamps-labs/yolo-stamp", "cpu") # тут указать device
    image = Image.open("images/img5.png")
    result = pipeline(image)
    for i in result.numpy():
        x1, y1, x2, y2 = i[0], i[1], i[2], i[3]
        stamp_crop = image.crop((x1, y1, x2, y2))
        stamp_crop.save(f"stamp_{i}.png")
