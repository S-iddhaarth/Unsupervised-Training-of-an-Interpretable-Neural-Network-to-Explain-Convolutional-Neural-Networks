from timm.models import create_model
import sys
sys.path.append(r'../')
# import VanillaNet.models.vanillanet
import torch
import utils.utility
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader

# import custom_dataloader
import os
import generator.network
import generator.feature_extractor
import utils.utility
# import custom_dataloader
from datasets import load_dataset
from collections import defaultdict

# vanillanet_model = create_model(
#         'vanillanet_5',
#         pretrained=False,
#         num_classes=1000,
#         act_num=3,
#         drop_rate=0,
#         deploy=False,
#         )
# pretrained = r"../models/vanillanet_5.pth"
# vanillanet_model.load_state_dict(torch.load(pretrained)['model'])
# vanillanet_model.to('cuda:1')

from torchvision.models import resnet18
model = resnet18(num_classes=10)
model.load_state_dict(torch.load(r'../weights/resnet18_checkpoints_MNIST/resnet18_epoch_4.pth'))
model.to('cuda')


layers = utils.utility.getConv(model)
print(layers)
hooks = generator.feature_extractor.FeatureExtractor(model,layers)

path = r'../feature_maps/MNIST_resnet_map'

if not os.path.exists(path):
        os.mkdir(path)

_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.485, 0.229)
])

# Data transformation function
def transform(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

# Load dataset with transforms
data = load_dataset('MNIST').with_transform(transform)
trainLoader = DataLoader(data['test'], batch_size=32,pin_memory=True, num_workers=4)


def _get_map(hook,image,layers):
    features = hook(image)
    map = torch.zeros((image.shape[0], len(layers),224,224), device='cuda')
    feature_number = 0
    for k in features.values():
        k = torch.mean(k, dim=1)
        k = utils.utility.resize_array(k, (224, 224))
        for l in range(k.shape[0]):
            map[l][feature_number] = k[l]
        feature_number += 1
    return map
counter = defaultdict(int)
for batch in trainLoader:
        image, gt = batch["pixel_values"].to('cuda'), batch["label"].to('cuda')
        
        activation_map = _get_map(hooks,image,layers)
        activation_map = activation_map.detach().cpu()
        counta = 0
        for class_ in gt:
                print(f'storing {class_} image {counter[class_]}')
                class_ = str(int(class_))
                dir = os.path.join(path,class_)
                if not os.path.exists(dir):
                        os.mkdir(dir)
                torch.save(activation_map[counta],os.path.join(dir,f'{str(counter[class_])}.pt'))
                counta += 1
                counter[class_] += 1