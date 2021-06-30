import csv
import pandas as pd

all = [1, 2, 3, 4, 5]
allcsv = "five_train_200.csv"

for key in all:
    csv_path = "%s.csv" % key
    file = pd.read_csv(csv_path, header=None)
    file = file.sample(frac=1)[:200]  #对每一个"%s.csv" % key,做全采样以打乱顺序，取前x个。
    with open(allcsv, 'a+', newline='') as output:
        for indexs in file.index:
            csv_write = csv.writer(output)
            csv_write.writerow(file.loc[indexs].values[:].T)
