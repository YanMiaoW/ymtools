import torch 
import numpy as np 

def tensor2mask(tensor):
    return (tensor[0]*255).cpu().numpy().astype(np.uint8)

def tensor2image(tensor):
    return ((tensor.permute(1, 2, 0)+1)*0.5*255).cpu().numpy().astype(np.uint8)
