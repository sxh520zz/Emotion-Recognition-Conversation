# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle
import math
import numpy as np
# reload a file to a variable
with open('../Feature/Text_data_No_combine.pickle', 'rb') as file:
    train_org_data_map = pickle.load(file)

Turn = 6

#print(len(train_org_data_map))
#print(train_org_data_map[0][0])

def remove_nested_list(listt):
    while [] in listt:  # 判断是否有空值在列表中
        listt.remove([])  # 如果有就直接通过remove删除
    return listt

def Pre_did(all_data):
    Train_data = []
    for i in range(len(all_data)):
        num = 0
        Train_data_1 = [[] for x in range(500)]
        Train_data_1[0].append(all_data[i][0])
        for j in range(1,len(all_data[i])):
            if(Train_data_1[num][-1]['id'][-4] == all_data[i][j]['id'][-4]):
                Train_data_1[num].append(all_data[i][j])
            else:
                num = num +1
                Train_data_1[num].append(all_data[i][j])
        Train_data_2 = remove_nested_list(Train_data_1)
        Train_data.append(Train_data_2)
    return Train_data

'''
def Class_data(all_data):
    Train_data = [[] for x in range(500)]
    for data_ind in all_data:
        Train_data[int(data_ind['Dialogue_ID'])].append(data_ind)

    train_data = []
    for i in range(len(Train_data)):
        dia_data = {}
        name_list = []
        for j in range(len(Train_data[i])):
            if (Train_data[i][j]['Speaker'] not in name_list):
                name_list.append(Train_data[i][j]['Speaker'])
        dia_data['Speaker_num'] = len(name_list)
        dia_data['Dia_length'] = len(Train_data[i])
        dia_data['Dia_data'] = Train_data[i]
        train_data.append(dia_data)

    CAN_USE = []
    num = 0
    for i in range(len(train_data)):
        if (train_data[i]['Dia_length'] >= 7):
            CAN_USE.append(train_data[i]['Dia_data'])
            num = num + train_data[i]['Dia_length']
    print(num)
    return CAN_USE
'''

def Design_emotion_change(train_map):
    no_change = 0
    change = 0
    a= []
    for i in range(len(train_map)):
        x_data = []
        for j in range(2,len(train_map[i])):
            if(len(train_map[i][j]) == 1):
                if(train_map[i][j][-1]['label'] == train_map[i][j -2][-1]['label']):
                    train_map[i][j][-1]['Self_Emotion_Change'] = 0
                    no_change = no_change+1
                    #x.append(train_org_data_map[i][j])
                else:
                    train_map[i][j][-1]['Self_Emotion_Change'] = 1
                    change = change + 1
                    #x.append(train_org_data_map[i][j])
                if(train_map[i][j][-1]['label'] == train_map[i][j -1][-1]['label']):
                    train_map[i][j][-1]['Inter_Emotion_Change'] = 0
                    no_change = no_change+1
                    #x.append(train_org_data_map[i][j])
                else:
                    train_map[i][j][-1]['Inter_Emotion_Change'] = 1
                    change = change + 1
            else:
                for x in range(len(train_map[i][j])):
                    if (x == 0):
                        if (train_map[i][j][x]['label'] == train_map[i][j - 2][-1]['label']):
                            train_map[i][j][x]['Self_Emotion_Change'] = 0
                            no_change = no_change + 1
                            # x.append(train_org_data_map[i][j])
                        else:
                            train_map[i][j][x]['Self_Emotion_Change'] = 1
                            change = change + 1
                            # x.append(train_org_data_map[i][j])
                        if (train_map[i][j][x]['label'] == train_map[i][j - 1][-1]['label']):
                            train_map[i][j][x]['Inter_Emotion_Change'] = 0
                            no_change = no_change + 1
                            # x.append(train_org_data_map[i][j])
                        else:
                            train_map[i][j][x]['Inter_Emotion_Change'] = 1
                            change = change + 1
                    else:
                        if (train_map[i][j][x]['label'] == train_map[i][j][x-1]['label']):
                            train_map[i][j][x]['Self_Emotion_Change'] = 0
                            no_change = no_change + 1
                            # x.append(train_org_data_map[i][j])
                        else:
                            train_map[i][j][x]['Self_Emotion_Change'] = 1
                            change = change + 1
                            # x.append(train_org_data_map[i][j])
                        if (train_map[i][j][x]['label'] == train_map[i][j - 1][-1]['label']):
                            train_map[i][j][x]['Inter_Emotion_Change'] = 0
                            no_change = no_change + 1
                            # x.append(train_org_data_map[i][j])
                        else:
                            train_map[i][j][x]['Inter_Emotion_Change'] = 1
                            change = change + 1
            x_data.append(train_map[i][j])
        a.append(x_data)
    return a

def Train_data(train_map):
    train_data_ALL_1 = []
    label_list= [1,2,3,4,5]
    num = 0
    for i in range(len(train_map)):
        train_data = []
        for j in range(len(train_map[i]) - Turn):
            if (len(train_map[i][j + Turn]) == 1):
                self_information = []
                inter_information = []
                context_information = []
                for x in range(Turn):
                    if(x % 2 == 0):
                        for y in range(len(train_map[i][j + x])):
                            self_information.append(train_map[i][j + x][y])
                            context_information.append(train_map[i][j + x][y])
                    else:
                        for y in range(len(train_map[i][j + x])):
                            inter_information.append(train_map[i][j + x][y])
                            context_information.append(train_map[i][j + x][y])
                data = {}
                data['self_information'] = self_information
                data['inter_information'] = inter_information
                data['context_information'] = context_information
                data['label'] = train_map[i][j + Turn][0]['label']
                data['Self_Emotion_Change'] = train_map[i][j + Turn][0]['Self_Emotion_Change']
                data['Inter_Emotion_Change'] = train_map[i][j + Turn][0]['Inter_Emotion_Change']
                data['id'] = train_map[i][j + Turn][0]['id']

                if(data['label'] in label_list):
                    if(data['label'] == 5):
                        data['label'] = 2
                    data['label'] = data['label'] - 1
                    train_data.append(data)
                    num = num + 1
            else:
                for w in range(len(train_map[i][j + Turn])):
                    if (w == 0):
                        self_information = []
                        inter_information = []
                        context_information = []
                        for x in range(Turn):
                            if (x % 2 == 0):
                                for y in range(len(train_map[i][j + x])):
                                    self_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                            else:
                                for y in range(len(train_map[i][j + x])):
                                    inter_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                        data = {}
                        data['self_information'] = self_information
                        data['inter_information'] = inter_information
                        data['context_information'] = context_information
                        data['label'] = train_map[i][j + Turn][0]['label']
                        data['Self_Emotion_Change'] = train_map[i][j + Turn][0]['Self_Emotion_Change']
                        data['Inter_Emotion_Change'] = train_map[i][j + Turn][0]['Inter_Emotion_Change']
                        data['id'] = train_map[i][j + Turn][0]['id']
                        if (data['label'] in label_list):
                            if (data['label'] == 5):
                                data['label'] = 2
                            data['label'] = data['label'] - 1
                            train_data.append(data)
                            num = num + 1
                    else:
                        self_information = []
                        inter_information = []
                        context_information = []
                        for x in range(Turn):
                            if (x % 2 == 0):
                                for y in range(len(train_map[i][j + x])):
                                    self_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                            else:
                                for y in range(len(train_map[i][j + x])):
                                    inter_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])

                        for wx in range(w):
                            self_information.append(train_map[i][j + Turn][wx])
                            context_information.append(train_map[i][j + Turn][wx])
                        data = {}
                        data['self_information'] = self_information
                        data['inter_information'] = inter_information
                        data['context_information'] = context_information
                        data['label'] = train_map[i][j + Turn][w]['label']
                        data['Self_Emotion_Change'] = train_map[i][j + Turn][w]['Self_Emotion_Change']
                        data['Inter_Emotion_Change'] = train_map[i][j + Turn][w]['Inter_Emotion_Change']
                        data['id'] = train_map[i][j + Turn][w]['id']
                        if (data['label'] in label_list):
                            if (data['label'] == 5):
                                data['label'] = 2
                            data['label'] = data['label'] - 1
                            train_data.append(data)
                            num = num + 1
        train_data_ALL_1.append(train_data)

    print(len(train_data_ALL_1))
    print(len(train_data_ALL_1[0]))
    print(len(train_data_ALL_1[0][0]))
    print(num)

    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []

    for i in range(len(train_data_ALL_1)):
        for j in range(len(train_data_ALL_1[i])):
            if (train_data_ALL_1[i][j]['id'][4] == '1'):
                data_1.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '2'):
                data_2.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4]== '3'):
                data_3.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '4'):
                data_4.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '5'):
                data_5.append(train_data_ALL_1[i][j])

    data = []
    data.append(data_1)
    data.append(data_2)
    data.append(data_3)
    data.append(data_4)
    data.append(data_5) 
    return data



train_map = Pre_did(train_org_data_map)
train_org_data_map = Design_emotion_change(train_map)
Train_data = Train_data(train_org_data_map)
file = open('Train_data.pickle', 'wb')
pickle.dump(Train_data, file)


