import torch as T
import torch.nn as nn
import os
import sys
import torchaudio
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim


sys.path.append("./")
from AudioLoader import AudioDataset

class GAN(nn.Module):
    def __init__(self, bCuda = True, g_lr = 0.00005, d_lr = 0.00005, epochs = 1000, bSaveProgressSamples = True, sample_rate = 16000, lenght_sec = 1, batch_size = 32, noise_dim = 100):
        super().__init__()
        self.epochs = epochs
        self.bSaveProgressSamples = bSaveProgressSamples

        self.device = 'cuda' if bCuda and T.cuda.is_available() else 'cpu'
        print(f"using: {self.device}")

        self.discr = Discriminator().to(self.device)
        self.gener = Generator(noise_dim).to(self.device)

        self.g_lr = g_lr
        self.d_lr = d_lr

        self.sample_rate = sample_rate
        self.lenght_sec = lenght_sec
        self.total_samples = lenght_sec * sample_rate
        self.batch_size = batch_size
        self.noise_dim = noise_dim


        self.optim_g = optim.Adam(self.gener.parameters(), lr=g_lr, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.discr.parameters(), lr=d_lr, betas=(0.5, 0.999))
        self.loss = nn.BCELoss()

        self.real_label = 0.9
        self.fake_label = 0.1
        self.fixed_noise = T.randn(1, self.noise_dim, device=self.device)
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
        DATA_PATH = "./Data"
        dataset = AudioDataset(
                data_dir = DATA_PATH,
                target_sample_rate=self.sample_rate,
                num_samples=self.total_samples
                )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train(self):
        dataset = self.get_dataset()
        for e in range(0, self.epochs):
            for idx, (batch, label) in enumerate(dataset):
                self.optim_d.zero_grad()

                noise = T.randn(batch.size(0), self.noise_dim, device=self.device)
                fake = self.gener(noise)
                fake = fake.detach()

                real_val = self.discr(batch).view(-1)
                fake_val = self.discr(fake).view(-1)

                label = torch.full((batch.shape[0],), self.real_label,dtype=T.float, device=self.device)
                real_loss = self.loss(real_val, label)
                real_loss.backward()

                label.fill_(self.fake_label)
                fake_losss = self.loss(fake_val, label)
                fake_losss.backward()

                self.optim_d.step()


                self.optim_g.zero_grad()
                label.fill_(self.real_label)
                noise = T.randn(batch.size(0), self.noise_dim, device=self.device)
                y = self.gener(noise)
                val = self.discr(y).view(-1)
                g_loss = self.loss(val, label)
                g_loss.backward()
                self.optim_g.step()

                print(f"epoch: {e}, batch: {idx}, fake val: {fake_val.mean().item()}, real val: {real_val.mean().item()}, real loss: {real_loss.mean.item()}, fake loss: {fake_losss.mean().item()}, gen loss: {g_loss.mean().item()}")




    def gen(self):
        self.gener.eval()
        with T.no_grad():
            noise = T.randn(1, self.noise_dim, device=self.device)
            wave = self.gener(noise)
        self.gener.train()
        return wave


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
    def __init__(self, noise_dim, out_chanels = 1):
        super().__init__()

        self.init_len = 125
        self.fc = nn.Linear(noise_dim, 256 * self.init_len)
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
            nn.BatchNorm1d(out_chanels),
            nn.LeakyReLU(0.2),

            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, self.init_len)
        return self.main(x)
