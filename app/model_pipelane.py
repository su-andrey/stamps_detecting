import albumentations as A
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from huggingface_hub import hf_hub_download
from detection_models.utils import output_tensor_to_boxes, nonmax_suppression, xywh2xyxy

class YoloStampPipeline:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2(p=1.0),
        ])

    @classmethod
    def from_pretrained(
            cls,
            model_path_hf: str = None,
            device: str = "cpu",
            filename_hf: str = "weights.pt",
            local_model_path: str = None
    ):
        yolo = cls(device)
        if model_path_hf is not None and filename_hf is not None:
            yolo.model = torch.load(hf_hub_download(model_path_hf, filename=filename_hf), map_location="cpu")
            yolo.model.to(yolo.device)
            yolo.model.eval()
        elif local_model_path is not None:
            yolo.model = torch.load(local_model_path, map_location="cpu")
            yolo.model.to(device)
            yolo.model.eval()
        return yolo

    def __call__(self, image) -> torch.Tensor:
        shape = torch.tensor(image.size)
        coef = torch.hstack((shape, shape)) / 448
        image = image.convert("RGB").resize((448, 448))
        image = np.array(image)
        image_tensor = self.transform(image=image)
        output = self.model(image_tensor["image"].unsqueeze(0).to(self.device))
        boxes = output_tensor_to_boxes(output[0].detach().cpu())
        boxes = nonmax_suppression(boxes=boxes)
        boxes = xywh2xyxy(torch.tensor(boxes)[:, :4])
        boxes = boxes * coef
        return boxes
