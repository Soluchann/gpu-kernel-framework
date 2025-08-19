import torch
import torch.nn.functional as F

def conv2d(input, weight, stride=1, padding=0):
    return F.conv2d(input, weight, stride=stride, padding=padding)
