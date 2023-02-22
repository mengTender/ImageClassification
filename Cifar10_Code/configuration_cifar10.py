# _*_ coding: utf-8 _*_
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms


models = {
    'VGG16': tv_models.vgg16(num_classes=10),
    'VGG16_pretained': tv_models.vgg16( num_classes=10),
    'VGG19': tv_models.vgg19(num_classes=10),
    'VGG19_pretained': tv_models.vgg19( num_classes=10),
    'ResNet18': tv_models.resnet18(num_classes=10),
    'ResNet18_pretrained': tv_models.resnet18(num_classes=10),
    'ResNet50': tv_models.resnet50(num_classes=10),
    'ResNet50_pretrained': tv_models.resnet50( num_classes=10)
}

transforms_dict = {
    'train_transform': transforms.Compose([
        transforms.RandomInvert(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'test_transform': transforms.ToTensor()
}
