from model import LeNet
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
epoch = 100
data_train = MNIST('./data',
                   download=False,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()
                   ]))
data_test = MNIST('./data',
                  train=False,
                   download=False,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()
                   ]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=1024,num_workers=0)

model = LeNet()
model.train()#模型切换到训练状态
lr  = 0.1 #学习率
criterion = nn.CrossEntropyLoss()      #定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                            weight_decay=5e-4)# 定义随机梯度下降优化器


for i in range(10):
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predict = outputs.max(1)
        total += targets.size(0)
        correct += predict.eq(targets).sum().item()
        print(batch_idx, len(data_train_loader),'Loss: %.3f | (Acc: %.3f %%(%d/%d'%(train_loss/(batch_idx+1),100.*correct/total,correct,total))

save_info = {
    #保存的信息
    "iter_num":epoch, #迭代的步数
    "optimizer": optimizer.state_dict(), #优化器的状态字典
    "model": model.state_dict(), #模型的状态字典
}
save_path = r"./model/model.pth"
torch.save(save_info, save_path)