import sys
import numpy as np
import scipy.io.wavfile
sys.path.append('./')
from gan import GAN as gan

class GanBase():
    def __init__(self):
        super().__init__()
        self.gan = gan()

    def train(self, bClean = False, bCuda = True, g_lr = 0.00005, d_lr = 0.00005, epochs = 1000, bSaveProgressSamples = True):
        self.gan = gan(bCuda = bCuda, g_lr = g_lr, d_lr = d_lr, epochs = epochs, bSaveProgressSamples = bSaveProgressSamples)

        if bClean:
            print(f"Train mode: clean: {bClean}, use cuda if available: {bCuda}, generator learning rate: {g_lr}, discriminator learning rate: {d_lr}, epochs: {epochs}, save progress samples: {bSaveProgressSamples}")
        else:
            self.gan.load()
            print ("loading longest trined model")
        self.gan.train()

    def gen(self, samples = 1):
        path = "../Output/"
        print(f"Gen mode: samples to generate: {samples}, paht: {path}")
        for i in range(0, samples):
            wave = self.gan.gen()
            wave_np = wave.squeeze().detach().cpu().numpy()
            max_val = np.max(np.abs(wave_np))
            if max_val > 0:
                wave_np = wave_np / max_val
            wave_int16 = (wave_np * 32767).astype(np.int16)
            
            scipy.io.wavfile.write(f"./Output/sample_{i}.wav", 16000, wave_int16)
