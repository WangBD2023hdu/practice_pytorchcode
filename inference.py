import torch
import torch.nn as nn
from model import LeNet
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

# save_info = {
#     #保存的信息
#     "iter_num":iter_num, #迭代的步数
#     "optimizer": optimizer.state_dict(), #优化器的状态字典
#     "model": model.state_dict(), #模型的状态字典
# }

model_path = "./model/model.pth" #
save_info = torch.load(model_path)
model = LeNet()
criterion = nn.CrossEntropyLoss()
model.load_state_dict(save_info["model"])
model.eval()

test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_test_loader):

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predict = outputs.max(1)
        total += targets.size(0)
        correct += predict.eq(targets).sum().item()
        print(batch_idx, len(data_train_loader), 'Loss: %.3f | (Acc: %.3f %%(%d/%d' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))