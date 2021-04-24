import os
import random
import paddle
from paddle.dataset.uci_housing import load_data
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import gzip
import json

paddle.seed(0)
random.seed(0)
np.random.seed(0)

# 数据文件
datafile = '../work/mnist.json.gz'  # 训练数据的存放地址
print('loading mnist dataset from {} ......'.format(datafile))  # 打印提示“正在从‘地址’加载‘训练数据名称’”
data = json.load(gzip.open(datafile))  # 将从所给训练数据的内容读取并保存到data中
train_set, val_set, eval_set = data  # 将data分成三个部分，训练集、验证集和测试集

# 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
IMG_ROWS = 28
IMG_COLS = 28
imgs, labels = train_set[0], train_set[1]  # 每个集包含两个部分，【0】是图片，是一个28*28的向量，【1】是标签，内容为这张图片所代表的数字
print("训练数据集数量: ", len(imgs))  # 打印训练数据集的数量，或者说行数
assert len(imgs) == len(labels), \
    "length of train_imgs({}) should be the same as train_labels({})".format(
        len(imgs), len(labels))  # 检测图片的个数和标签的个数是否相等，如果不相等就报错

from paddle.io import Dataset


class MnistDataset(Dataset):
    def __init__(self):
        self.IMG_COLS = 28
        self.IMG_ROWS = 28

    def __getitem__(self, idx):
        image = train_set[0][idx]
        image = np.array(image)
        image = image.reshape((1, IMG_ROWS, IMG_COLS)).astype('float32')  # 将之前的image重新整理为只有一行，每个元素为28*28二维数组，
        # 数据类型为32浮点的array数组
        label = train_set[1][idx]
        label = np.array(label)
        label = label.astype('int64')  # 标签重新整理为一行的整型array
        return image, label

    def __len__(self):
        return len(imgs)


# 调用加载数据的函数
dataset = MnistDataset()
train_loader = paddle.io.DataLoader(dataset, batch_size=100, shuffle=True, return_list=True)


# 定义模型结构
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化层卷积核kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化层卷积核kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是10
        self.fc = Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    def forward(self, inputs, label):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        x = F.softmax(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = False
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

EPOCH_NUM = 5
BATCH_SIZE = 100


def train(model):
    model.train()

    BATCH_SIZE = 100
    # 定义学习率，并加载优化器参数到模型中
    total_steps = (int(50000 // BATCH_SIZE) + 1) * EPOCH_NUM
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
    # 使用Adam优化器
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())

    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            image_data = data[0].reshape([BATCH_SIZE, 1, 28, 28])  # 这里的1代表通道数，这里使用的数据是1通道，常用RGB是3通道
            label_data = data[1].reshape([BATCH_SIZE, 1])
            image = paddle.to_tensor(image_data)
            label = paddle.to_tensor(label_data)
            # if batch_id<10:
            # print(label.reshape([-1])[:10])
            # 前向计算的过程
            predict, acc = model(image, label)
            avg_acc = paddle.mean(acc)
            # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.cross_entropy(predict, label)
            avg_loss = paddle.mean(loss)

            # 每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            avg_acc.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

            # 保存模型参数和优化器的参数
            paddle.save(model.state_dict(), '../checkpoint/mnist_epoch{}'.format(epoch_id) + '.pdparams')
            paddle.save(opt.state_dict(), '../checkpoint/mnist_epoch{}'.format(epoch_id) + '.pdopt')
    print(opt.state_dict().keys())


def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = '../checkpoint/mnist_epoch4.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_loader = load_data('eval')

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        predicts, acc = model(images, labels)
        loss = F.cross_entropy(input=predicts, label=labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    # 计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))


model = MNIST()

train(model)

evaluation(model)

print(model.state_dict().keys())
