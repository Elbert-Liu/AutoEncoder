from torch.nn import functional as F
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas
import matplotlib.pyplot as plt
import torch

BATCH_SIZE = 64
NUM_EPOCHS = 200


class MyDataset(Dataset):  # 读取csv文件，第0列是label，之后的是image，返回image，label。

    def __init__(self, csv_file, transform=None):
        self.data_df = pandas.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        image = self.data_df.iloc[index, 1:].values
        if self.transform:
            image = torch.FloatTensor(image) / 255
        return image, label

    def __len__(self):
        return len(self.data_df)


train_dataset = MyDataset('five_train_200.csv', transform=True)  # 指定训练数据集
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 数据集加载


# 深度自动编码机的类，包括encoder和decoder两个结构。return mid与out
class deep_auto_encoder(nn.Module):
    def __init__(self):
        super(deep_auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 250),
            nn.LeakyReLU(),
            nn.Linear(250, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 250),
            nn.LeakyReLU(),
            nn.Linear(250, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 28 * 28)
        )

    def forward(self, x):
        mid = self.encoder(x)
        out = self.decoder(mid)
        return mid, out


net = deep_auto_encoder()  # 网络加载
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 优化器选择Adam,学习率降到0.0001

train_loss = []

for epoch in range(NUM_EPOCHS):  # 训练过程
    for i, data in enumerate(train_dataloader):
        images = data[0]
        labels = data[1]
        mid, out = net(images)
        loss = F.mse_loss(out, images)  # out与images的loss函数选择mse-loss

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 权重优化

        train_loss.append(loss.item())  # 记录训练的loss值
        print(epoch + 1, loss.item())  # 简单展示训练的loss值


test_dataset = MyDataset('five_test.csv', transform=True)  # 指定训练数据集
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 加载训练数据集
net.eval()  # 模型固定
plot_1_x = []  # 绘图时的x_list,为mid的所有第0维数据
plot_1_y = []  # 绘图时的y_list,为mid的所有第1维数据
plot_2_x = []  # 绘图时的x_list,为mid的所有第0维数据
plot_2_y = []  # 绘图时的y_list,为mid的所有第1维数据
plot_3_x = []  # 绘图时的x_list,为mid的所有第0维数据
plot_3_y = []  # 绘图时的y_list,为mid的所有第1维数据
plot_4_x = []  # 绘图时的x_list,为mid的所有第0维数据
plot_4_y = []  # 绘图时的y_list,为mid的所有第1维数据
plot_5_x = []  # 绘图时的x_list,为mid的所有第0维数据
plot_5_y = []  # 绘图时的y_list,为mid的所有第1维数据

count = 0
for i, data in enumerate(test_dataloader):  # 测试过程
    images = data[0]
    labels = data[1]
    mid, out = net(images)
    # tensor转numpy，否则无法被matplotlib加载
    X, Y = mid[:, 0].detach().numpy(), mid[:, 1].detach().numpy()

    for label in labels:
        if label == 1:
            plot_1_x.append(X[count])
            plot_1_y.append(Y[count])
            count = count+1
        elif label == 2:
            plot_2_x.append(X[count])
            plot_2_y.append(Y[count])
            count = count+1
        elif label == 3:
            plot_3_x.append(X[count])
            plot_3_y.append(Y[count])
            count = count+1
        elif label == 4:
            plot_4_x.append(X[count])
            plot_4_y.append(Y[count])
            count = count+1
        else:
            plot_5_x.append(X[count])
            plot_5_y.append(Y[count])
            count = count+1

axes = plt.subplot(111)
one = axes.scatter(plot_1_x, plot_1_y, c='b')  # 散点图
two = axes.scatter(plot_2_x, plot_2_y, c='g')  # 散点图
three = axes.scatter(plot_3_x, plot_3_y, c='r')  # 散点图
four = axes.scatter(plot_4_x, plot_4_y, c='c')  # 散点图
five = axes.scatter(plot_5_x, plot_5_y, c='m')  # 散点图


axes.legend((one, two, three, four, five),
            (u'one', u'two', u'three', u'four', u'five'), loc=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("The result of 200 numbers of each type in the training set")
plt.savefig("train_five_200.png")
plt.show()
