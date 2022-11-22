# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle
import math
import numpy as np
# reload a file to a variable
with open('Train_data_org.pickle', 'rb') as file:
    Train_data_org = pickle.load(file)
with open('Train_data.pickle', 'rb') as file:
    Train_data = pickle.load(file)

data = []
num = 0
for i in range(len(Train_data_org)):
    data_1 = []
    for j in range(len(Train_data_org[i])):
        for x in range(len(Train_data)):
            for y in range(len(Train_data[x])):
                if(Train_data_org[i][j]['id'] == Train_data[x][y]['id']):
                    data_1.append(Train_data[i][j])
                    num = num + 1
    data.append(data_1)
print(num)

file = open('Train_data_Old.pickle', 'wb')
pickle.dump(data, file)


