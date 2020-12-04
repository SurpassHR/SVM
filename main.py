# -*- coding: utf-8 -*-
# @Time : 2020/12/1 16:23
# @Author : KevinHoo
# @Site : 
# @File : main.py
# @Software: PyCharm 
# @Email : hu.rui0530@gmail.com


# 用于分析的库
import pandas as pd
import numpy as np
from sklearn import svm

# 用于可视化的库
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)  # 字体缩放比例

# 读取写好的配料表csv文件
recipes = pd.read_csv('./data/muffin_or_cupcakes_real.csv')
recipes.head(5)
# print(recipes)

# 先绘制出其中的两种配料即二维的数据图，因为糖和黄油占比及不算太多，也不算太少
# params: 横纵坐标, 数据集, 分类变量, 画板色调, 显示数据范围, 画板样式参数s for mark_size
sns.lmplot(x='Sugar', y='Butter', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})

# 拟合模型
# 按照指定的模型输入
sugar_butter = recipes[['Sugar', 'Butter']].values
type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)

# SVM模型
# model = svm.SVC(kernel='linear', C=2**-5)  # 支持向量分类器，内核设置为线性
model = svm.SVC(kernel='linear', C=1, gamma=1e10)  # 支持向量分类器，内核设置为线性
model.fit(sugar_butter, type_label)

# 得到分割的超平面wtx+b=0，但由于是二维的，他的超平面为二维减去一维就类似于y=ax+b
w = model.coef_[0]
# print(model.coef_)
# 有ax+(1/b)y+c=0, y=-abx-cb
a = -w[0] / w[1]
xx = np.linspace(5, 30)  # 返回(5, 30)的等间距样本
# print(xx)
yy = a * xx - (model.intercept_[0] / w[1])
# print(yy)

# 画出穿过支持向量表示超平面范围(路宽)的虚线
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])  # 下方虚线，过支持向量  y-y0=k(x-x0)
b = model.support_vectors_[-3]
yy_up = a * xx + (b[1] - a * b[0])  # 上方虚线，过支持向量

# 分别查看超平面和边界
sns.lmplot(x='Sugar', y='Butter', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')  # 相同的x取值范围，绘制超平面
plt.plot(xx, yy_down, 'k--')  # 相同的x取值范围，绘制下边界虚线
plt.plot(xx, yy_up, 'k--')  # 相同的x取值范围，绘制上边界虚线
# model.support_vectors_ = model.support_vectors_[:]，类似于列表的数据存储方式
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=80, facecolors='none')


# 绘制测试配料的数据点，查看点的位置
sns.lmplot(x='Sugar', y='Butter', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={'s': 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(12, 12, 'yo', markersize='9')


# 输出判断结果
def muffin_or_cupcake(butter, sugar):
    if(model.predict([[butter, sugar]])) == 0:
        print('\n这是松糕的配料表！')
    else:
        print('\n这是纸杯蛋糕的配料表！')


muffin_or_cupcake(12, 12)

# C参数，分辨并避免被误分类的数据点
# 用一个低的C值拟合模型
model = svm.SVC(kernel='linear', C=2**-5)
model.fit(sugar_butter, type_label)
# 用一个高的C值拟合模型
model = svm.SVC(kernel='linear', C=2**5)
model.fit(sugar_butter, type_label)




