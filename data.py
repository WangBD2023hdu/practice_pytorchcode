from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = MNIST('./data',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()
                   ]))
data_test = MNIST('./data',
                  train=False,
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()
                   ]))

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=1024,num_workers=0)

import matplotlib.pyplot as plt

figure = plt.figure()
num_of_images = 60
for imgs, target in data_train_loader:
    break

for index in range(num_of_images):
    plt.subplot(6, 10, index + 1)
    plt.axis('off')
    img = imgs[index,:]
    plt.imshow(img.numpy().squeeze(), cmap='gray_r')    #squeeze 把张量维度为1的阶去掉
plt.show()
