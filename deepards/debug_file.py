import torch
import torch.nn as nn
import torch.nn.functional as F

model = torch.load("pretrained-model-1.pth",map_location=torch.device('cpu'))
print(model)
   