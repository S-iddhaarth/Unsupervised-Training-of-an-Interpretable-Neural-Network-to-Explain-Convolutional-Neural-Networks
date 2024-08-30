import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed import init_process_group
import numpy as np
import random
import os
from collections import defaultdict


def get_numeric_part(filename):
    # Split the filename into numeric and non-numeric parts
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part)  # Convert numeric part to integer for sorting

def sort_filenames_by_number(filenames):
    # Sort filenames based on the numeric part extracted from each filename
    sorted_filenames = sorted(filenames, key=get_numeric_part)
    return sorted_filenames

def resize_array(array, new_shape):
    array = array.unsqueeze(1)
    return F.interpolate( array, size=new_shape, mode='bilinear', align_corners=False)


def linear_stretching_channelwise(input_tensor, a=0.0, b=1.0):
    num_channels, height, width = input_tensor.shape
    reshaped_tensor = input_tensor.view(num_channels, -1)  # Shape: (num_channels, height * width)

    # Calculate min and max values for each channel
    min_vals = torch.min(reshaped_tensor, dim=1)[0]  # Min values across each channel
    max_vals = torch.max(reshaped_tensor, dim=1)[0]  # Max values across each channel
    print(reshaped_tensor.shape)
    # Expand min_vals and max_vals to match the original tensor shape
    min_vals = min_vals.view(num_channels, 1, 1)  # Shape: (num_channels, 1, 1)
    max_vals = max_vals.view(num_channels, 1, 1)  # Shape: (num_channels, 1, 1)

    # Apply linear stretching to each channel separately
    stretched_tensor = (input_tensor - min_vals) * (b - a) / (max_vals - min_vals) + a

    return stretched_tensor

def getConv(model :nn.Module)->list:
  layers = []
  for layer_id, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        layers.append(layer_id)
  return layers

def grad_flow_dict(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            
    return {layers[i]: ave_grads[i] for i in range(len(ave_grads))}

def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def ddp_setup(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend='nccl',rank=rank,world_size=world_size)
    
def deNormalize(image,device):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    denormalized_images = image * std + mean
    denormalized_images = torch.clamp(denormalized_images, 0, 1)
    return denormalized_images

counter = defaultdict(int)
def saveVccMap(root,vcc,extractor,image,label):
    
    if not os.path.exists(root):
        os.mkdir(root)
    maps = extractor(image)
    batch = image.shape[0]
    channels = len(maps)

    features = torch.zeros((batch,channels,224,224))
    feature_number = 0
    for feat in maps.values():
            with torch.no_grad():
                feat = vcc(feat)
                feat = feat.squeeze(1)
                feat = resize_array(feat,(224,224))
                feat = feat.squeeze(1)
                features[:,feature_number,:,:] = feat
            feature_number += 1

    for save in range(batch):
        _class = str(int(label[save]))
        dir = os.path.join(root,_class)
        if not os.path.exists(dir):
            os.mkdir(dir)
        torch.save(features[save].detach().cpu(),os.path.join(dir,f'{counter[_class]}.pt'))
        torch.save(image[save].detach().cpu(),os.path.join(dir,f'i{counter[_class]}.pt'))
        print(f'{_class} - {counter[_class]}')
        counter[_class] += 1
        
def get_map_vcc(hook,vcc,image,layers):
    features = hook(image)
    map = torch.zeros((image.shape[0], len(layers),224,224))
    feature_number = 0
    for feat in features.values():
        with torch.no_grad():
            feat = vcc(feat)
            feat = feat.squeeze(1)
            feat = resize_array(feat,(224,224))
            feat = feat.squeeze(1)
            map[:,feature_number,:,:] = feat
        feature_number += 1
    return map

def get_map(hook,image,layers):
    features = hook(image)
    map = torch.zeros((image.shape[0], len(layers),224,224))
    feature_number = 0
    for k in features.values():
        k = torch.mean(k, dim=1)
        k = resize_array(k, (224, 224))
        for l in range(k.shape[0]):
            map[l][feature_number] = k[l]
        feature_number += 1
    return map

counter_map = defaultdict(int)

def save_map(image,gt,hook,layers,path):
    if not os.path.exists(path):
        os.mkdir(path)
    image = image.to('cuda')
    gt = gt.to('cuda')
    activation_map = get_map(hook,image,layers)
    activation_map = activation_map.detach().cpu()
    counta = 0
    for class_ in gt:
        print(f'storing {class_} image {counter[class_]}')
        class_ = str(int(class_))
        dir = os.path.join(path,class_)
        if not os.path.exists(dir):
                os.mkdir(dir)
        torch.save(activation_map[counta].detach().cpu(),os.path.join(dir,f'{str(counter_map[class_])}.pt'))
        torch.save(image[counta].detach().cpu(),os.path.join(dir,f'i{str(counter_map[class_])}.pt'))
        counta += 1
        counter_map[class_] += 1

def apply_mask(image,mask):
    mask = mask.repeat(1,3,1,1)
    return torch.mul(image,mask)