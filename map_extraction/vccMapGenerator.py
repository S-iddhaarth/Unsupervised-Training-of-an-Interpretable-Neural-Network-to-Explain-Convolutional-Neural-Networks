import generator.network
import torch
from torch import nn
import json
from torchvision.transforms import transforms
import map_image_loader
from torch.utils.data import DataLoader
import generator.feature_extractor
import utils
from timm.models import create_model
import sys
sys.path.append(r'../')
import VanillaNet.models.vanillanet
import os
from collections import defaultdict

vanillanet_model = create_model(
            'vanillanet_5',
            pretrained=False,
            num_classes=1000,
            act_num=3,
            drop_rate=0,
            deploy=False,
            )

pretrained = r"../models/vanillanet_5.pth"
vanillanet_model.load_state_dict(torch.load(pretrained)['model'])
vanillanet_model.to('cuda')

vcc = generator.network.ExplanationNetwork(1).to('cuda')
vcc.load_state_dict(torch.load(r'/home/explainable-AI/classification/changes_1/gen4.pth'))
vcc = vcc.fmp_layerwise

layers = utils.utility.getConv(vanillanet_model)[0:-2]
extractor = generator.feature_extractor.FeatureExtractor(vanillanet_model,layers).to('cuda')

with open('config.json','r') as fl:
    config = json.load(fl)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((255,255)),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
    # Add any other transforms you need
    ])
dataset = map_image_loader.CustomTensorDataset('valid',config['map'],transform=transform)
train_dataloader = DataLoader(dataset, batch_size=10)

root = r'vcc_valid_map'
if not os.path.exists(root):
    os.mkdir(root)
counter = defaultdict(int)
c = 0

for i,j,k in train_dataloader:
    maps = extractor(i.to('cuda'))
    del i
    del j
    features = torch.zeros((10,8,224,224))
    feature_number = 0
    c += 1
    print(f'c - {c}')
    if c > 0:
        for feat in maps.values():
            with torch.no_grad():
                feat = vcc(feat)
                feat = feat.squeeze(1)
                feat = utils.utility.resize_array(feat,(224,224))
                feat = feat.squeeze(1)
                features[:,feature_number,:,:] = feat
            feature_number += 1 

        for save in range(10):
            _class = str(int(k[save]))
            dir = os.path.join(root,_class)
            if not os.path.exists(dir):
                os.mkdir(dir)
            torch.save(features[save].detach().cpu(),os.path.join(dir,f'{counter[_class]}.pt'))
            print(f'{_class} - {counter[_class]}')
            counter[_class] += 1
