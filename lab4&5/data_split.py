# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/4 4:34 下午
@Auth ： Codewyf
@File ：data_split.py
@IDE ：PyCharm
@Motto：Go Ahead Instead of Heasitating

"""
# kdd99数据集预处理
# 将kdd99符号型数据转化为数值型数据
# coding:utf-8
# kdd99数据集预处理
# 将kdd99符号型数据转化为数值型数据
# coding:utf-8
import numpy as np
import pandas as pd
import csv
import time

global label_list  # label_list为全局变量


# 定义kdd99数据预处理函数
def preHandel_data():
    source_file = 'car.csv'
    handled_file = 'corrected5.csv'
    data_file = open(handled_file, 'w', newline='')  # python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        csv_writer = csv.writer(data_file)
        count = 0  # 记录数据的行数，初始化为0
        for row in csv_reader:
            row = row[0].split(',');
            temp_line = np.array(row)  # 将每行数据存入temp_line数组里
            temp_line[1] = handleOverPrice(row)  # 将源文件行中3种协议类型转换成数字标识
            temp_line[2] = handleBuyPrice(row)  # 将源文件行中70种网络服务类型转换成数字标识
            temp_line[3] = handleDoors(row)  # 将源文件行中11种网络连接状态转换成数字标识
            temp_line[4] = handlePersons(row)  # 将源文件行中23种攻击类型转换成数字标识
            temp_line[5] = handleLug_boot(row)
            temp_line[6] = handleLabel(row)
            # temp_line[6] = handleLabel(row)

            csv_writer.writerow(temp_line)
            count += 1
            # 输出每行数据中所修改后的状态
            print(count, 'status:', temp_line[1], temp_line[2], temp_line[3], temp_line[41])
        data_file.close()


# 将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
def find_index(x, y):
    return [i for i in range(len(y)) if y[i] == x]


# 定义将源文件行中3种协议类型转换成数字标识的函数
def handleOverPrice(input):
    Overprice_list = ['vhigh', 'high', 'med','low']
    if input[1] in Overprice_list:
        return find_index(input[1], Overprice_list)[0]


# 定义将源文件行中70种网络服务类型转换成数字标识的函数
def handleBuyPrice(input):
    BuyPrice_list = ['vhigh', 'high', 'med','low']
    if input[2] in BuyPrice_list:
        return find_index(input[2], BuyPrice_list)[0]


# 定义将源文件行中11种网络连接状态转换成数字标识的函数
def handleDoors(input):
    Doors_list = ['2','3','4','5more']
    if input[3] in Doors_list:
        return find_index(input[3], Doors_list)[0]

def handlePersons(input):
    Persons_list = ['2','4','more']
    if input[4] in Persons_list:
        return find_index(input[4], Persons_list)[0]

def handleLug_boot(input):
    Lug_boot_list = ['small','med','big']
    if input[5] in Lug_boot_list:
        return find_index(input[5], Lug_boot_list)[0]

def handleLabel(input):
    Label_list = ['uncacc','acc','good','vgood']
    if input[5] in Label_list:
        return find_index(input[5], Label_list)[0]


# 定义将源文件行中攻击类型转换成数字标识的函数(训练集中共出现了22个攻击类型，而剩下的17种只在测试集中出现)
# def handleLabel(input):
#
#     global label_list  # 在函数内部使用全局变量并修改它
#     if input[6] in label_list:
#         return find_index(input[6], label_list)[0]
#     else:
#         label_list.append(input[6])
#         return find_index(input[6], label_list)[0]


if __name__ == '__main__':
    start_time = time.time();
    global label_list  # 声明一个全局变量的列表并初始化为空
    label_list = []
    preHandel_data()
    end_time = time.time();

    print("Running time:", (end_time - start_time))  # 输出程序运行时间

