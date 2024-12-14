from dino_v2_detect.Module.detector import Detector

def demo():
    model_file_path = '/home/chli/Model/DINOv2/dinov2_vitg14_reg4_pretrain.pth'
    device = 'cpu'
    image_file_path = '/home/chli/Dataset/CapturedImages/y_5_x_3.png'

    detector = Detector(model_file_path, device)

    dino_feature = detector.detectFile(image_file_path)

    print('dino_feature:')
    print(dino_feature)
    print(dino_feature.shape)
    return True
