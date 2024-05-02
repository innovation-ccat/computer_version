import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5
from utils.Functions import get_accuracy, plot_losses

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# parameters
RANDOM_SEED = 24  # 随机数种子
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 64
Epochs = 30
N_CLASSES = 10


def train(train_loader, model, loss_func, optimizer, device):
    """
    定义单次训练过程

    :param train_loader:训练数据流
    :param model:网络模型
    :param loss_func:损失函数
    :param optimizer:优化器
    :param device:使用GPU or CPU
    :return:模型状态，优化器状态，本次训练损失
    """

    model.train()  # 切换训练模式
    running_loss = 0

    for imgs, labels in train_loader:
        optimizer.zero_grad()  # 优化器梯度初始化

        imgs = imgs.to(device)
        labels = labels.to(device)

        # 前向传播
        output, prob = model(imgs)
        loss = loss_func(output, labels)  # 计算损失
        running_loss += loss.item() * imgs.size(0)

        # 反向传播
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, loss_func, device):
    """

    :param valid_loader:验证集数据流
    :param model:网络模型
    :param loss_func:损失函数
    :param device:GPU or CPU
    :return:模型状态，本次验证损失
    """

    model.eval()  # 设置为验证模式
    running_loss = 0

    for imgs, labels in valid_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass and record loss
        output, prob = model(imgs)
        loss = loss_func(output, labels)
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_process(model, loss_func, optimizer, train_loader, valid_loader, epochs, device):
    """
    完整训练过程
    :param model:网络模型
    :param loss_func:损失函数
    :param optimizer:优化器
    :param train_loader:训练集数据流
    :param valid_loader:验证集数据流
    :param epochs:训练次数
    :param device:GPU or CPU
    :param print_every:每隔多少次训练打印
    :return:模型状态，优化器,(训练损失，验证损失)
    """
    # 记录最佳结果
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    for epoch in range(0, epochs):
        # 训练
        start_time = time.time()
        model, optimizer, train_loss = train(train_loader, model, loss_func, optimizer, device)
        train_losses.append(train_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_model = model
        # 验证
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, loss_func, device)
            valid_losses.append(valid_loss)

        # 在验证集上计算准确度
        valid_acc = get_accuracy(model, valid_loader, device=device)
        end_time = time.time()
        print("Epoch:%d    TrainLoss=%.4f    ValidLoss=%.4f    ValidAccuracy=%.2f%%    Time cost=%.2fs" % (
        epoch, train_loss, valid_loss, valid_acc * 100, end_time - start_time))
    # 绘制loss曲线
    print("**********Minimum Loss is in Epoch%d = %.4f**********" % (best_epoch, best_loss))
    plot_losses(train_losses, valid_losses)

    return best_model, optimizer, (train_losses, valid_losses)


transforms = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# download and create datasets
train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='data',
                               train=False,
                               transform=transforms)

# define the data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

if __name__ == '__main__':
    print(DEVICE)
    torch.manual_seed(RANDOM_SEED)
    model = LeNet5(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model, optimizer, _ = training_process(model, criterion, optimizer, train_loader, valid_loader, Epochs, DEVICE)
    torch.save(model, "Model/LeNet5.pth")
