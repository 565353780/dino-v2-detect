import os
import torch
import numpy as np
from PIL import Image
from typing import Union
from torchvision import transforms

from dino_v2_detect.Model.vision_transformer import vit_giant2


class Detector(object):
    def __init__(self, model_file_path: Union[str, None]=None, device: str = 'cpu') -> None:
        self.device = device
        # self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.dtype = torch.float16

        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

        self.model = vit_giant2(
            patch_size=14,
            num_register_tokens=4,
            img_size=518,
            ffn_layer='swiglufused',
            block_chunks=0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict, strict=True)
        return True

    @torch.no_grad()
    def detect(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(image.cpu().numpy())

        image = image.convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)

        dino_feature = self.model(image_tensor)

        return dino_feature

    @torch.no_grad()
    def detectFile(self, image_file_path: str) -> Union[torch.Tensor, None]:
        if not os.path.exists(image_file_path):
            print('[ERROR][Detector::detectFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return None

        image = Image.open(image_file_path)

        dino_feature = self.detect(image)

        return dino_feature
