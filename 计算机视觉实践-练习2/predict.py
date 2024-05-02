import torch
from PIL import Image
from cv2 import cv2
from matplotlib import pyplot as plt
from numpy.random import choice
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('Model/LeNet5.pth')  # 加载模型
model = model.to(device)
model.eval()  # 把模型转为test模式


def multi_predict():
    """
    在MNIST上随机抽取若干数据识别
    :return: None
    """
    trans = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # 选择50张看效果
    col = 10
    row = 5
    test_dataset = datasets.MNIST(root='data', train=False, transform=trans)

    fig = plt.figure()

    test_dataset_cnt = len(test_dataset)
    test_index = [i for i in range(test_dataset_cnt)]
    random_index = choice(test_index, row * col, replace=False)  # 随机挑选图片

    for i, index in enumerate(random_index):
        sample = test_dataset[index]
        img = sample[0]  # 源图像tensor数据

        img_show = img[0][:, :]  # 供显示的图片
        img_show = img_show.numpy()

        img = img.to(device)  # 输入模型的图片
        img_input = img.unsqueeze(0)

        label = sample[1]  # 真实标签
        plt.subplot(col, row, i + 1)
        plt.axis('off')
        plt.imshow(img_show, cmap='gray')

        with torch.no_grad():
            _, probs = model(img_input)

        title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
        plt.title(title, fontsize=7)
    fig.suptitle('Predictions')
    plt.show()


def single_predict(img_path):
    """
    单张图片检测
    :param img_path:图片路径
    :return:label
    """
    img = cv2.imread(img_path)  # 读取要预测的图片
    img_show = img.copy()
    trans = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图
    img = img / 255  # opencv读取的灰度图片是0-255，映射到0-1之间
    img = 1 - img  # 反相
    img = Image.fromarray(img)  # 这里ndarray_image为原来的numpy数组类型的输入

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # [1，1，28，28]
    with torch.no_grad():
        _, probs = model(img)
    title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
    plt.imshow(img_show, cmap='gray')
    plt.title(title)
    plt.show()
    return f'{torch.argmax(probs)}'


if __name__ == '__main__':
    # multi_predict()
    label = single_predict("test/6.png")
    print("Predicted result is: ", label)
