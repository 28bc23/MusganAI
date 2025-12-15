class GanBase():
    def __init__(self):
        super().__init__()

    def train(self, bClean = False, bCuda = True, g_lr = 0.00005, d_lr = 0.00005, epochs = 1000, bSaveProgressSamples = True):
        if bClean:
            print(f"Train mode: clean: {bClean}, use cuda if available: {bCuda}, generator learning rate: {g_lr}, discriminator learning rate: {d_lr}, epochs: {epochs}, save progress samples: {bSaveProgressSamples}")
        else:
            print ("loading longest trined model")

    def gen(self, samples = 1):
        print("Gen mode: samples to generate: {samples}")
