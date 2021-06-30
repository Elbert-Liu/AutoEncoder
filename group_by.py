import csv
from itertools import groupby

for key, rows in groupby(csv.reader(open("mnist_test.csv")), lambda row: row[0]):
    with open("%s.csv" % key, 'a+', newline='') as output:
        for row in rows:
            csv_write = csv.writer(output)
            csv_write.writerow(row)