# -*- coding: utf-8 -*-
# @Time : 2020/12/1 11:30
# @Author : KevinHoo
# @Site : 
# @File : test2.py
# @Software: PyCharm 
# @Email : hu.rui0530@gmail.com

# encoding=utf-8

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # sklearn.cross_validation在新版本替换为model_selection
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm


def check_wrong_item(train_data):
    # 读取数据
    train = train_data

    # 检查数据中是否有缺失值，以下两种方式均可
    # Flase:对应特征的特征值中无缺失值
    # True：有缺失值
    print("缺失值\n", train.isnull().any())
    print(np.isnan(train).any())

    # 查看缺失值记录
    train_null = pd.isnull(train)
    train_null = train[train_null == True]
    print(train_null)

    # 缺失值处理，以下两种方式均可
    # 删除包含缺失值的行
    train.dropna(inplace=True)
    # 缺失值填充
    train.fillna('100')
    # 检查是否包含无穷数据
    # False:包含
    # True:不包含
    print(np.isfinite(train).all())
    # False:不包含
    # True:包含
    print(np.isinf(train).all())

    # 数据处理
    train_inf = np.isinf(train)
    train[train_inf] = 0


if __name__ == '__main__':

    print('prepare datasets...')
    # Iris数据集
    # iris=datasets.load_iris()
    # features=iris.data
    # labels=iris.target

    # MINST数据集

    raw_data = pd.read_csv('./data/train_data.csv', header=0, error_bad_lines=False)  # 读取csv数据，并将第一行视为表头，返回DataFrame类型
    print(np.isnan(raw_data).any())  # 报错提示有缺失值或无穷大的值，检测并打印空值部分，False无缺失值，True有
    raw_data.dropna(inplace=True)  # 替换空值部分
    # check_wrong_item(raw_data)
    data = raw_data.values
    features = data[::, 1::]
    labels = data[::, 0]    # 选取33%数据作为测试集，剩余为训练集

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

    time_2=time.time()
    print('Start training...')
    clf = svm.SVC()  # svm class
    clf.fit(train_features, train_labels)  # training the svc model
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict=clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
print("The accruacy score is %f" % score)
