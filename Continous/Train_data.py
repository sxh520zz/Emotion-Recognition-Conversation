# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle
import math
import numpy as np
# reload a file to a variable
with open('Text_data.pickle', 'rb') as file:
    train_org_data_map = pickle.load(file)

step = 16

def Design_emotion_change(train_org_data_map):
    no_change = 0
    change = 0
    a= []
    for i in range(len(train_org_data_map)):
        x = []
        for j in range(2,len(train_org_data_map[i])):
            if(train_org_data_map[i][j]['label'] == train_org_data_map[i][j -2]['label']):
                train_org_data_map[i][j]['Self_Emotion_Change'] = 0
                no_change = no_change+1
                #x.append(train_org_data_map[i][j])
            else:
                train_org_data_map[i][j]['Self_Emotion_Change'] = 1
                change = change + 1
                #x.append(train_org_data_map[i][j])
            if(train_org_data_map[i][j]['label'] == train_org_data_map[i][j -1]['label']):
                train_org_data_map[i][j]['Inter_Emotion_Change'] = 0
                no_change = no_change+1
                x.append(train_org_data_map[i][j])
            else:
                train_org_data_map[i][j]['Inter_Emotion_Change'] = 1
                change = change + 1
                x.append(train_org_data_map[i][j])


        a.append(x)

    print(no_change)
    print(change)
    print(no_change+change)
    return a

def Train_data(train_map):

    num = 0
    input_traindata_x = []
    input_traindata_y = []
    input_traindata_z = []
    input_traindata_x_1 = []
    input_traindata_x_2 = []
    input_traindata_y_1 = []
    for i in range(len(train_map)):
        input_trainlabel_1 = []
        input_trainlabel_2 = []
        input_trainlabel_3 = []
        input_traindata_3 = []
        input_traindata_4 = []
        input_traindata_5 = []
        for x in range(len(train_map[i]) - step):
            input_train_tran_1 = []
            input_train_trad_1 = []
            input_train_name_1 = []
            for y in range(step):
                b = train_map[i][x + y]['transcr_data']
                c = train_map[i][x + y]['trad_data']
                d = train_map[i][x + y]['id']
                input_train_tran_1.append(b)
                input_train_trad_1.append(c)
                input_train_name_1.append(d)
            input_trainlabel_1.append(train_map[i][x + step]['Self_Emotion_Change'])
            input_trainlabel_2.append(train_map[i][x + step]['label'])
            input_trainlabel_3.append(train_map[i][x + step]['Inter_Emotion_Change'])
            input_traindata_3.append(input_train_trad_1)
            input_traindata_4.append(input_train_name_1)
            input_traindata_5.append(input_train_tran_1)
            num = num+1
        input_traindata_x_1.append(input_trainlabel_1)
        input_traindata_x.append(input_trainlabel_2)
        input_traindata_x_2.append(input_trainlabel_3)
        input_traindata_y.append(input_traindata_3)
        input_traindata_z.append(input_traindata_4)
        input_traindata_y_1.append(input_traindata_5)
    print(num)


    label_list= [1,2,3,4,5]
    num = 0
    traindata_1 = []
    for i in range(len(input_traindata_z)):
        input_traindata_1_1 = []
        for x in range(len(input_traindata_z[i])):
            a = {}
            if (input_traindata_x[i][x] in label_list):
                #if (input_traindata_x_1[i][x] != input_traindata_x[i][x]):
                if (input_traindata_x[i][x] == 5):
                    input_traindata_x[i][x] = 2
                #a['label_emotion_change'] = int(input_traindata_x_1[i][x])
                a['label_emotion_change'] = int(input_traindata_x_2[i][x])
                a['label_emotion'] = int(input_traindata_x[i][x] - 1)
                a['trad_data'] = input_traindata_y[i][x]
                a['transcr_data'] = input_traindata_y_1[i][x]
                a['id'] = input_traindata_z[i][x]
                a['section_id'] = input_traindata_z[i][x][0][4]
                input_traindata_1_1.append(a)
                num = num + 1
        traindata_1.append(input_traindata_1_1)
    print(num)
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []

    for i in range(len(traindata_1)):
        for j in range(len(traindata_1[i])):
            if (traindata_1[i][j]['section_id'] == '1'):
                data_1.append(traindata_1[i][j])
            if (traindata_1[i][j]['section_id'] == '2'):
                data_2.append(traindata_1[i][j])
            if (traindata_1[i][j]['section_id'] == '3'):
                data_3.append(traindata_1[i][j])
            if (traindata_1[i][j]['section_id'] == '4'):
                data_4.append(traindata_1[i][j])
            if (traindata_1[i][j]['section_id'] == '5'):
                data_5.append(traindata_1[i][j])
    '''
    print(len(data_1))
    print(len(data_2))
    print(len(data_3))
    print(len(data_4))
    print(len(data_5))
    '''


    data = []
    data.append(data_1)
    data.append(data_2)
    data.append(data_3)
    data.append(data_4)
    data.append(data_5)
    return data

train_org_data_map = Design_emotion_change(train_org_data_map)
Train_data = Train_data(train_org_data_map)
file = open('Train_data.pickle', 'wb')
pickle.dump(Train_data, file)