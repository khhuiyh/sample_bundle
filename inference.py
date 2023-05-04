import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from multiprocessing import cpu_count
class SuperSimpleClassifier(nn.Module):
    def __init__(self):
        super(SuperSimpleClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=224)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(-1, 3)
        x = self.fc(x)
        return x
class Solution:
    def __init__(self, lr_scheduler_gamma=0.1):
        self.per = 4
        self.num_cpu = cpu_count()//2
        # self.num_cpu = 4
        self.is_parallel = 'pool1'
        # self.model = models.resnet18(pretrained=False)
        # self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model = SuperSimpleClassifier()
        self.is_train = False
        self.pretrained_name = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=lr_scheduler_gamma)
        self.epoch = 2
        self.batch_size = 8
        self.transform_train = transforms.Compose([
        transforms.Resize(256),  # 先将短边缩放到256
        transforms.CenterCrop(224),  # 再从中心裁剪出224x224大小的图像
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # 将图像转换为Tensor格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        self.transform_test = transforms.Compose([
        transforms.Resize(256),  # 先将短边缩放到256
        transforms.CenterCrop(224),  # 再从中心裁剪出224x224大小的图像
        transforms.ToTensor(),  # 将图像转换为Tensor格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])


    def train(self, x, y):
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return pred, loss.item()

    def test(self, x, y):
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
        return pred, loss.item()
    
    def inference(self, x):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
        return np.array(pred.argmax(dim=1)).reshape(-1)



    
