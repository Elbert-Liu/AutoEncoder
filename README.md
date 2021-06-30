## 说明文档
使用自动编码机和PCA完成对mnist数据集的降维可视化
```
模块需求
    python3
    pytorch
    pandas
    matplotlib
    numpy
    sklearn
数据集需求
	mnist_test.csv
	包括10000条数据，第0列为标签，之后784列为图片展平之后的数据。
```

#### 步骤一：将总数据集划分为10类，命名为x.csv

```
python group_by.py
```

#### 步骤二：将标签为1，2，3，4，5的csv文件，抽取指定数量的样本，命名为train_five_x.csv

```
python re_group_by.py
```

#### 步骤三：使用不同数量的数据集，去训练Deep_autoencoder，并使用five_test.csv测试，five_test.csv由每一类的10条数据组成

```
python Deep_autoencoder.py
```

#### 步骤四：直接使用PCA算法，将five_test.csv降到2维

```
python PCA.py
```

