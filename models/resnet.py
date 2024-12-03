### resnet.py
## 모델 정의만 담당하는 파일

### Resnet18 model
import torch.nn as nn
import torchvision.models as models

def create_resnet18(num_classes=6, pretrained=True):
    """
    ResNet18 모델 생성 함수

    Args:
        num_classes (int): 출력 클래스 수
        pretrained (bool): 사전 학습된 가중치 사용 여부

    Returns:
        nn.Module: ResNet18 모델
    """
    resnet18 = models.resnet18(pretrained=pretrained)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    return resnet18
