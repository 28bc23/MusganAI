import torch as T
import torch.nn as nn
import os
import sys
import torchaudio
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

sys.path.append("./")
from AudioLoader import AudioDataset

class GAN(nn.Module):
    def __init__(self, bCuda, g_lr, d_lr, epochs, bSaveProgressSamples):
        super().__init__()
        self.epochs = epochs
        self.bSaveProgressSamples = bSaveProgressSamples

        self.device = 'cuda' if bCuda and T.cuda.is_available() else 'cpu'
        print(f"using: {self.device}")

        self.discr = Discriminator().to(self.device)
        self.gener = Generator().to(self.device)

        self.g_lr = g_lr
        self.d_lr = d_lr

    def load(self):
        path = "../Models/InTraining/"
        biggest_epoch = -1
        longest_trained = None
        for p in os.scandir(path):
            name = os.path.splitext(os.path.basename(p.path))[0]
            epoch = name.split("_")[2]
            if epoch > biggest_epoch:
                biggest_epoch = epoch
                longest_trained = p 
        if longest_trained != None:
            epoch = save.get('epoch', 0)
            generator = save.get('generator', None)
            discriminator = save.get('discriminator', None)
            optim_g = save.get('optim_g', None)
            optim_d = save.get('optim_d', None)
            if (generator is not None) and (discriminator is not None) and (optim_g is not None) and (optim_d is not None):
                self.gener.load_state_dict(generator)
                self.discr.load_state_dict(discriminator)
                print("--- loaded ---")
                return True
            
        return False

    def save(self, epoch):
        path = f"../Models/InTraining/"
        if not os.path.exists(path):
            os.makedirs(path)

        save = {
            'epoch': epoch,
            'generator': self.gener.state_dict(),
            'discriminator': self.discr.state_dict(),
            'optim_g': self.optim_g.state_dict(),
            'optim_d': self.optim_d.state_dict(),
        }

        torch.save(save, f"{path}/musgan_epoch_{epoch}.pth")
        print("--- saved ---")

    def get_dataset(self):
        DATA_PATH = "../Data"
        dataset = AudioDataset(
                data_dir = DATA_PATH,
                target_sample_rate=16000,
                num_samples=16000 * 60
                )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        return dataloader

    def train(self):
        pass
    def gen(self):
        pass
    def optim(self):
        pass

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
        

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
