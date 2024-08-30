import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import wandb
import nvidia_smi
import copy
import sys
import time
import os
import gc
sys.path.append('../')

import generator.network
import generator.feature_extractor
import utils.utility
from loss.CrossEntropy import CrossEntropyLossMultiClassPercent
import torchvision.transforms as transforms
import map_image_loader
class trainer:
    def __init__(self, config: dict, model: torch.nn.Module,
                log: bool = False, project_name: str = None,
                resume: bool = False, seed: int = None,
                temperature: int = 0, sz: int = 224,
                img_save: bool = False):

        self.seed = seed
        if self.seed:
            utils.utility.seed_everything(self.seed)
            print(f'set seed to {self.seed}')
    
        self.img_save = img_save
        self.sz = sz
        self.config = config
        self.temperature = temperature
        self.cost = CrossEntropyLossMultiClassPercent(self.temperature).to(config['device'])

        print('initialized Loss function')
        self.log = log
        self.project_name = project_name
        self.resume = resume
        self.model = model
            
        print(f'Successfully registered hooks')
    
        print(f'initializing generator')
        self.generator = generator.network.ExplanationNetwork(self.config['channels']).to(config['device'])
        print('initializing optimizer')
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config['learning_rate'])
        self.running_loss = None
        self.running_mask = None

        self.train_data = self.loadDataset()

        if self.log:
            wandb.init(
                project=self.project_name,
                config={
                    "seed": self.seed,
                    "batch size": self.config["batch_size"],
                    "number of epochs": self.config["epochs"],
                    "learning rate": self.config["learning_rate"],
                }
            )

    def _mask_image(self, mask, input):
        
        """Takes mask (B,1,H,W) and image (B,C,H,W) as input and performs element-wise
        multiplication on eac
        """
        
        return torch.mul(mask.repeat(1, 3, 1, 1), input)


    def _run_batch(self, image,map, gt,step):
        
        self.optimizer.zero_grad()
        print('generated map')
        image = image.to(self.config['device'])
        map = map.to(self.config['device'])
        gt = gt.to(self.config['device'])
        mask = self.generator(image, map)
        print('generated mask')
        maskedImage = self._mask_image(mask, image)
        print(f'classification of masked image')
        maskedImageClassification = self.model(maskedImage)
        imageClassification = self.model(image)
        print(f'original - {gt} pred - {imageClassification.argmax(1)} masked - {maskedImageClassification.argmax(1)}')
        print('classifying image')
        
        loss, maskedImagePred_gt_loss, originalImagePred_gt_loss = self.cost(maskedImageClassification, imageClassification, gt)
        del maskedImageClassification
        del imageClassification
        gc.collect()
        loss = loss.to(self.config['device'])
            
        loss.backward()
        print('backpropagating loss')
        self.optimizer.step()
        print('stepping the optimizer')
        self.running_loss += loss.detach().item()
        self.running_mask += maskedImagePred_gt_loss.detach().item()
        print(f'\ttotal loss - {loss}\n\tMasked image ground truth error - {maskedImagePred_gt_loss}\n\toriginal image ground truth loss - {originalImagePred_gt_loss}\n\trunning_loss - {self.running_loss}')
        gradientFlow = utils.utility.grad_flow_dict(self.generator.named_parameters())
        if self.log:
            d = {"masked image - ground truth loss": maskedImagePred_gt_loss,
                "image - ground truth loss": originalImagePred_gt_loss,
                "loss": loss}
            d.update(gradientFlow)
            wandb.log(d)
        if self.img_save:
            if step%1000 == 0 or step == 20:
                imagesss = wandb.Image(mask, caption="Top: Output, Bottom: Input")
                wandb.log({'image':imagesss})

    def _run_epoch(self, epoch):
        nvidia_smi.nvmlInit()
        self.running_loss = 0
        self.running_mask = 0
        step = 0
        total = len(self.train_data)
        epochStart = time.time()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for data in self.train_data:
            step += 1
            print(f'....................step - {step} of {total}....................\n')
            batchStart = time.time()
            self._run_batch(data[0], data[1],data[2],step)
            for i in range(deviceCount):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100 * info.free / info.total, info.total, info.free, info.used))
            batchEnd = time.time()
            print(f'\taverage_loss - {self.running_loss / step}\n\tbatch time - {batchEnd - batchStart}')
        epochEnd = time.time()

        wandb.log({
            "loss per epoch": self.running_loss / len(self.train_data),
            "mask loss per epoch": self.running_mask / len(self.train_data),
            "epoch time": epochEnd - epochStart
        })
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1}, Loss: {self.running_loss / len(self.train_data):.4f}\ntime - {epochEnd - epochStart}")
        self._save_checkpoint(10,epoch)
        nvidia_smi.nvmlShutdown()

    def saveValidation(self, flname):
        with torch.no_grad():
            gc.collect()
            torch.cuda.empty_cache()
            self.validation = self.validation.to(self.config["device"])
            map1 = self._get_map(self.validation,0).to(self.config["device"])

            mask1 = self.generator(self.validation,map1)
            
        for i in range(mask1.shape[0]):
            plt.imsave(f'{flname}_{i}_{0}.png', mask1.to('cpu').detach().numpy()[i, 0, :, :], cmap='jet')

    def loadDataset(self, preprocessed=False):
        """
        Load datasets and return data loaders.

        Initializes and returns DataLoaders for training and (if logging is enabled) validation datasets.

        Returns:
            tuple: Contains DataLoaders. If logging is enabled:
                (train_loader, valid_test_img).
                Otherwise:
                (train_loader).
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((255,255)),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
    # Add any other transforms you need
        ])
        dataset = map_image_loader.ImageMapLoader(self.config['dataset'],self.config['map'],transform=transform,sample_per_class=2)
        train_dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

        return train_dataloader

    def _save_checkpoint(self, n, epoch):
        torch.save(self.generator.state_dict(), os.path.join(self.config['generator_model_path'], 'gen' + str(epoch % n) + ".pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.config['generator_model_path'], 'opt' + str(epoch % n) + ".pth"))

    def train(self):
        for epoch in range(self.config['epochs']):
            self.optimizer.zero_grad()
            self._run_epoch(epoch)