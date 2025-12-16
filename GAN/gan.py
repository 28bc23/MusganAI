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
                num_samples=16000 * 1
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
    def __init__(self, in_channel = 1):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv1d(in_channel, 64, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(0.2),

                nn.Conv1d(64, 128, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(0.2),

                nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(0.2),

                nn.Conv1d(256, 512, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(0.2),

                nn.Conv1d(512, 1024, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(0.2),
                )
        self.fc = nn.Linear(1024*16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)
        
        

class Generator(nn.Module):
    def __init__(self, in_size = 100, out_chanels = 1):
        super().__init__()

        self.init_len = 125
        self.fc = nn.Linear(in_size, 256 * self.init_len)
        self.main = nn.Sequential(
            
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(32, out_chanels, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, self.init_len)
        return self.main(x)
