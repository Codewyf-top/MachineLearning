import numpy as np
import pandas as pd
import csv
import time
global label_list  #label_list为全局变量


def handle():
    source_file = 'car.csv'
    handle_file = 'car2.csv'
    data_file = open(handle_file, 'w', newline='')
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        csv_writer = csv.writer(data_file)
        count = 0
        for row in csv_reader:
            #row = row[0].split(',')
            temp_line = np.array(row)
            temp_line[0] = buying(row)
            temp_line[1] = maint(row)
            temp_line[2] = doors(row)
            temp_line[3] = persons(row)
            temp_line[4] = lug_boot(row)
            temp_line[5] = safety(row)
            temp_line[6] = value(row)
            csv_writer.writerow(temp_line)
        data_file.close()


def findindex(x, y):
    return [i for i in range(len(y)) if y[i] == x]


def buying(flag1):
    list1 = ['vhigh', 'high', 'med', 'low']
    list1b = ['3', '2', '1', '0']
    if flag1[0] in list1:
        index = findindex(flag1[0], list1)[0]
        flag1[0] = list1b[index]
        return flag1[0]


def maint(flag1):
    list1 = ['vhigh', 'high', 'med', 'low']
    list1b = ['3', '2', '1', '0']
    if flag1[1] in list1:
        index = findindex(flag1[1], list1)[0]
        flag1[1] = list1b[index]
        return flag1[1]



def doors(flag1):
    if flag1[2] == '5more':
        flag1[2] = '5'
        return flag1[2]
    else:
        return flag1[2]


def persons(flag1):
    if flag1[3] == 'more':
        flag1[3] = '5'
        return flag1[3]
    else:
        return flag1[3]


def lug_boot(flag1):
    list2 = ['small', 'med', 'big']
    list2b = ['0', '1', '2']
    if flag1[4] in list2:
        index = findindex(flag1[4], list2)[0]
        flag1[4] = list2b[index]
        return flag1[4]


def safety(flag1):
    list3 = ['low', 'med', 'high']
    list3b = ['0', '1', '2']
    if flag1[5] in list3:
        index = findindex(flag1[5], list3)[0]
        flag1[5] = list3b[index]
        return flag1[5]


def value(flag1):
    list4 = ['unacc', 'acc', 'good', 'vgood']
    list5 = ['0', '1', '2', '3']
    if flag1[6] in list4:
        index = findindex(flag1[6], list4)[0]
        flag1[6] = list5[index]
        return flag1[6]


if __name__ == '__main__':
    handle()