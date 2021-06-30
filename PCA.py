import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 64

class MyDataset(Dataset):  # 读取csv文件，第0列是label，之后的是image，返回image，label。

    def __init__(self, csv_file, transform=None):
        self.data_df = pandas.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        image = self.data_df.iloc[index, 1:].values
        if self.transform:
            image = torch.FloatTensor(image)
        return image, label

    def __len__(self):
        return len(self.data_df)

def PCA_svd(X,k):
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k)
    pca.fit(X_std)
    x_reduction = pca.transform(X_std)

    return x_reduction

test_dataset = MyDataset('five_test.csv', transform=True)  # 指定训练数据集
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 加载训练数据集
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
    components = PCA_svd(images,2)
    X, Y = components[:, 0], components[:, 1]

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
plt.title("The result of PCA")
plt.savefig("PCA.png")
plt.show()