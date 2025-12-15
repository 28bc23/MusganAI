import torch as T
import torch.nn as nn

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
        pass

    def save(self):
        pass

    def get_dataset(self):
        pass

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
