# src/models/MultimodalFusion.py
import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "Time-Series-Library"))

from models.iTransformer import Model as iTransformerBase

class MultimodalFusionNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ts_model = iTransformerBase(configs)
        
        self.global_resnet = models.resnet18(pretrained=True)
        self.global_resnet.fc = nn.Linear(512, 2)
        
        self.local_resnet = models.resnet18(pretrained=True)
        self.local_resnet.fc = nn.Linear(512, 2)

    def forward(self, x_ts, x_global_img, x_local_img):
        ts_logits = self.ts_model(x_ts, None, None, None)
        global_logits = self.global_resnet(x_global_img)
        local_logits = self.local_resnet(x_local_img)
        return (ts_logits + global_logits + local_logits) / 3