import sys

sys.path.append('./GAN/')
import GanBase as Base


TRAIN_MODE = 1
GEN_MODE = 2
 
base = Base.GanBase()


print("select mode")
print(" 1. train")
print(" 2. generate")
mode = int(input("mode: "))
print (f"selected mode: {mode}")

if mode == TRAIN_MODE:
    cleanStr = input("Do you wanna load longest trained model? (Y/n): ")
    if cleanStr.lower() == "n":
        bCuda = True
        g_lr = 0.0005
        d_lr = 0.0005
        epochs = 1000
        bSaveProgressSample = True


        bCudaStr = input("Do you wanna use Cuda if available? (Y/n): ")
        if bCudaStr.lower() == "n":
            bCuda = False

        g_lr_str = input("enter learning rate for generator (0.0005): ")
        if g_lr_str.isnumeric():
            g_lr = g_lr_str

        d_lr_str = input("enter learning rate for discriminator (0.0005): ")
        if d_lr_str.isnumeric():
            d_lr = d_lr_str

        epochs = int(input("epochs to train (int): "))
 
        progressStr = input("Wanna save samples to track progress (Y/n): ")
        if progressStr.lower() == "n":
            bSaveProgressSample = False

        base.train(bClean = True, bCuda = bCuda, g_lr = g_lr, d_lr = d_lr, epochs = epochs, bSaveProgressSamples = bSaveProgressSample)

    else:
        base.train(bClean = False)


elif mode == GEN_MODE:
    base.gen()
else:
    print("else")
