import os

import torch
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image#读图片
import os.path
import time
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image#读图片


#数据读取与预处理操作
data_dir=r'E:\LX\GIT\git\test\对比实验\vit-pytorch\data\卫星云图1'
train_dir=data_dir+'train'
test_dir=data_dir+'test'

#制作好数据源
#data_transforms中指定了所有图像预处理的操作
#ImageFolder假设所有的文件夹保存好，每个文件夹下面储存同一类别的图片，文件夹的名字为分类的名字
#下面是数据增强操作
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomRotation(45),  #随机旋转，-45 ~ 45度之间随机选
        transforms.CenterCrop(227),  #从中心开始随机裁剪
        # transforms.RandomHorizontalFlip(p=0.5), #随机水平翻转，选择一个概率
        # transforms.RandomVerticalFlip(p=0.5),  #随机垂直翻转，选择一个概率
        # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), #随机调整亮度，对比度，饱和度和色相
        transforms.RandomGrayscale(p=0.025), #随机将图片转换为灰度图，3通道就是 R=G=B
        transforms.ToTensor(), #将图片转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #归一化，均值，标准差
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),  #改变图片大小
        transforms.CenterCrop(227),  #从中心开始随机裁剪
        transforms.ToTensor(),  #将图片转为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#加载数据集，转成小批次数据
batch_size=16
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir ,x),data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'test']}
dataset_size= {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels  # 下一层的输入通道数
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 使用2*2的池化层，步幅为2
    return nn.Sequential(*layers)  # 将所有层打包


def vgg(conv_arch, num_classes=11):
    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    net = nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(2048, num_classes)
    )
    return net

# VGG-19 的配置
conv_arch_vgg19 = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
num_classes = 11  # 根据你的数据集类别数量调整

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = vgg(conv_arch_vgg19, num_classes=num_classes)

def evaluate_accuracy_gpu(model, data_iter, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return correct / total, all_preds, all_labels

def train(net, train_loader, test_loader, num_epochs, lr, device):
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # 记录训练准确率和损失
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)

        # 测试集评估
        test_acc, true_labels, pred_labels = evaluate_accuracy_gpu(net, test_loader, device)
        test_accuracies.append(test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {(correct_train/total_train):.4f}, Test Accuracy: {test_acc:.4f}')

    # 绘制训练损失、训练准确率和测试准确率变化曲线
    plt.figure(figsize=(12, 5))
    plt.plot(range(num_epochs), train_losses, label='Train Loss', color='blue')
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy', color='green')
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy', color='orange')
    plt.title('Training Loss and Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # 混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化处理

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['Dersert', 'ExtratropicalCyclone', 'FrontalSurface', 'HighIceCloud', 'LowWaterCloud', 'Ocean', 'PatchElse', 'Snow', 'TropicalCyclone', 'Vegetation','WesterlyJet'])
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax)
    # 调整标签显示方式
    plt.xticks(rotation=90, verticalalignment='center', horizontalalignment='right')
    plt.yticks(rotation=0)
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    plt.show()

    # 打印分类报告，包含每个类别的精确度、召回率、F1分数和支持度
    report = classification_report(true_labels, pred_labels, target_names=['Dersert', 'ExtratropicalCyclone', 'FrontalSurface', 'HighIceCloud', 'LowWaterCloud', 'Ocean', 'PatchElse', 'Snow', 'TropicalCyclone', 'Vegetation','WesterlyJet'], digits=4)
    print("Classification Report:\n", report)

# x=torch.randn(size=(1,1,224,224))
# for blk in net:
#     x=blk(x)
#     print(blk.__class__.__name__,'output shape:\t',x.shape)

# # 训练模型
# lr, num_epochs, batch_size = 0.05, 10, 128
# train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
# train(net, train_iter, test_iter, num_epochs, lr, device=device)
if __name__ == '__main__':
    batch_size=16
    #训练集
    train_iter,test_iter=dataloaders['train'],dataloaders['test']
    #训练
    train(net,train_iter,test_iter,230,0.01,device=device)