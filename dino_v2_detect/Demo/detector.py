import torch
from tqdm import trange

from dino_v2_detect.Module.detector import Detector

def demo():
    model_type = 'base'
    model_file_path = '/home/chli/chLi/Model/DINOv2/dinov2_vitb14_reg4_pretrain.pth'
    dtype = 'auto'
    device = 'cuda'
    image_file_path = '/home/chli/chLi2/Dataset/CapturedImage/ShapeNet/02691156/10155655850468db78d106ce0a280f87/y_5_x_3.png'

    detector = Detector(model_type, model_file_path, dtype, device)

    for _ in trange(100):
        dino_feature = detector.detect(torch.randn([3, 3, 518, 518]))

    for _ in trange(100):
        dino_feature = detector.detectFile(image_file_path)

    print('dino_feature:')
    print(dino_feature)
    print(dino_feature.shape)
    return True
