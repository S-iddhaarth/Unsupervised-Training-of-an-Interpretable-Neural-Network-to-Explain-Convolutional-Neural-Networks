import trainer
from timm.models import create_model
import sys
sys.path.append(r'../')
import VanillaNet.models.vanillanet
import torch
import json
import torchvision.models as models


def main():
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
    vanillanet_model.switch_to_deploy()
    vanillanet_model.to('cuda')
    
  
    with open('config.json','r') as fl:
        conf = json.load(fl)

    training = trainer_singleGPU_corrected.trainer(config=conf,model=vanillanet_model,
                                                   log=True,project_name="Full model training",
                                                   seed = 100,temperature=0.3,img_save=True)
    training.train()

if __name__ == "__main__":
    main()