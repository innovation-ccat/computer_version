import torch
import  os
import time
from matplotlib import pyplot as plt


def get_accuracy(model, data_loader, device):
    """
    计算在数据集上的精确度
    :param model:网络模型
    :param data_loader:数据集
    :param device:GPU or CPU
    :return:Accuracy
    """
    correct = 0
    n = 0

    with torch.no_grad():
        model.eval()  # 验证模式
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            output, y_prob = model(imgs)
            _, predicted_labels = torch.max(y_prob, 1)  # 预测的标签
            n += labels.size(0)
            correct += (predicted_labels == labels).sum()

    return correct.float() / n  # 预测正确的/总共的


def plot_losses(train_losses, valid_losses):
    """
    绘制训练和验证过程中的损失情况
    :param train_losses:训练损失list
    :param valid_losses:验证损失List
    :return:None
    """
    # 创建 'result' 目录
    if not os.path.exists("result"):
        os.makedirs("result")

    plt.plot(train_losses, color='blue', label='Training loss')
    plt.plot(valid_losses, color='red', label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    # 生成唯一的文件名，基于当前时间戳
    timestamp = int(time.time())
    file_name = f"loss_plot_{timestamp}.png"
    save_path = os.path.join("result", file_name)

    plt.savefig(save_path)  # 保存图像到 result 目录下
    plt.show()

