"""
@author: Guo Shiguang
@software: PyCharm
@file: myplot.py
@time: 2022/4/4 15:29
"""

import csv

import matplotlib.pyplot as plt
import numpy


def smooth(scalar, weight=0.999):
    smoothed = []
    last = scalar[0]
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


csv_reader = list(csv.reader(open(r"C:\Users\gsg18\Documents\本科毕设\中期\valid.csv")))[1:]
valid_pre = list(map(list, zip(*csv_reader)))
valid = [list(map(int, valid_pre[0])), list(map(float, valid_pre[1])),
         list(map(float, valid_pre[2]))]

# print(valid)

csv_reader = list(csv.reader(open(r"C:\Users\gsg18\Documents\本科毕设\中期\train.csv")))[1:]
train_pre = list(map(list, zip(*csv_reader)))

train = [list(map(int, train_pre[0])), list(map(float, train_pre[1])),
         list(map(float, train_pre[2]))]

train_loss = smooth(train[1])
train_f1 = smooth(train[2])
valid_loss = smooth(valid[1])
valid_f1 = smooth(valid[2])

xs = numpy.linspace(0, 22050, 22050)
plt.plot(train[0], train_loss)
# plt.plot(xs, train_f1)
plt.plot(valid[0], valid_loss)
# plt.plot(xs, valid_f1)
plt.title("Smooth Spline Curve")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
