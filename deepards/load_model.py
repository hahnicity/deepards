import torch

model = torch.load('densenet-linear-kf5-model--fold-1.pth')

#for param_tensor in model.state_dict():
 #   print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print(model.modules)