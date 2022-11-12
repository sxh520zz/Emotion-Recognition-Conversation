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

Turn = 6
def remove_nested_list(listt):
    while [] in listt:  # 判断是否有空值在列表中
        listt.remove([])  # 如果有就直接通过remove删除
    return listt

def Get_data(data,tag):
    Out_data_0 = []
    Out_data_1 = []
    Out_data_2 = []
    for i in range(len(data['self_information'])):
        Out_data_0.append(data['self_information'][i][tag])
    for i in range(len(data['inter_information'])):
        Out_data_1.append(data['inter_information'][i][tag])
    for i in range(len(data['context_information'])):
        Out_data_2.append(data['context_information'][i][tag])
    Out_data = [Out_data_0,Out_data_1,Out_data_2]
    return Out_data

def Design_Turn(train_org_data):
    Turn_Data = []
    for i in range(len(train_org_data)):
        data_turn_level = [[] for _ in range(500)]
        Turn_utt = 0
        for x in range(len(train_org_data[i])):
            if (x == 0):
                data_turn_level[Turn_utt].append(train_org_data[i][x])
            elif(train_org_data[i][x]['id'][-4] == data_turn_level[Turn_utt][0]['id'][-4]):
                data_turn_level[Turn_utt].append(train_org_data[i][x])

            else:
                Turn_utt = Turn_utt + 1
                data_turn_level[Turn_utt].append(train_org_data[i][x])
        Turn_Dia = remove_nested_list(data_turn_level)
        Turn_Data.append(Turn_Dia)
    return Turn_Data

def Design_emotion_change(train_org_data_map):
    no_change = 0
    change = 0
    for i in range(len(train_org_data_map)):
        for j in range(2,len(train_org_data_map[i])):
            if(len(train_org_data_map[i][j]) == 1):
                if(train_org_data_map[i][j][0]['label'] == train_org_data_map[i][j -2][-1]['label']):
                    train_org_data_map[i][j][0]['Self_Emotion_Change'] = 0
                    no_change = no_change+1
                else:
                    train_org_data_map[i][j][0]['Self_Emotion_Change'] = 1
                    change = change + 1
                if(train_org_data_map[i][j][0]['label'] == train_org_data_map[i][j -1][-1]['label']):
                    train_org_data_map[i][j][0]['Inter_Emotion_Change'] = 0
                    no_change = no_change+1
                else:
                    train_org_data_map[i][j][0]['Inter_Emotion_Change'] = 1
                    change = change + 1
            else:
                for y in range(len(train_org_data_map[i][j])):
                    if(y == 0):
                        if (train_org_data_map[i][j][y]['label'] == train_org_data_map[i][j - 2][-1]['label']):
                            train_org_data_map[i][j][y]['Self_Emotion_Change'] = 0
                            no_change = no_change + 1
                        else:
                            train_org_data_map[i][j][y]['Self_Emotion_Change'] = 1
                            change = change + 1
                        if (train_org_data_map[i][j][y]['label'] == train_org_data_map[i][j - 1][-1]['label']):
                            train_org_data_map[i][j][y]['Inter_Emotion_Change'] = 0
                            no_change = no_change + 1
                        else:
                            train_org_data_map[i][j][y]['Inter_Emotion_Change'] = 1
                            change = change + 1
                    else:
                        if (train_org_data_map[i][j][y]['label'] == train_org_data_map[i][j][y-1]['label']):
                            train_org_data_map[i][j][y]['Self_Emotion_Change'] = 0
                            no_change = no_change + 1
                        else:
                            train_org_data_map[i][j][y]['Self_Emotion_Change'] = 1
                            change = change + 1
                        if (train_org_data_map[i][j][y]['label'] == train_org_data_map[i][j - 1][-1]['label']):
                            train_org_data_map[i][j][0]['Inter_Emotion_Change'] = 0
                            no_change = no_change + 1
                        else:
                            train_org_data_map[i][j][y]['Inter_Emotion_Change'] = 1
                            change = change + 1

    train_data = []
    for i in range(len(train_org_data_map)):
        train_data.append(train_org_data_map[i][2:])

    num = 0
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            for x in range(len(train_data[i][j])):
                num = num +1
    print(num)
    #print(no_change)
    #print(change)
    #print(no_change+change)
    return train_data

def Train_data(train_map):
    num = 0
    train_data_ALL = []
    for i in range(len(train_map)):
        train_data = []
        for j in range(len(train_map[i]) - Turn):
            if(len(train_map[i][j + Turn]) == 1):
                self_information = []
                inter_information = []
                context_information = []
                for x in range(Turn):
                    for y in range(len(train_map[i][j + x])):
                        if (x%2 == 0):
                            self_information.append(train_map[i][j + x][y])
                            context_information.append(train_map[i][j + x][y])
                        else:
                            inter_information.append(train_map[i][j + x][y])
                            context_information.append(train_map[i][j + x][y])
                data = {}
                data['self_information'] = self_information
                data['inter_information'] = inter_information
                data['context_information'] = context_information
                data['label'] = train_map[i][j + Turn][0]
                train_data.append(data)
                num = num +1
            else:
                for w in range(len(train_map[i][j + Turn])):
                    if(w == 0):
                        self_information = []
                        inter_information = []
                        context_information = []
                        for x in range(Turn):
                            for y in range(len(train_map[i][j + x])):
                                if (x % 2 == 0):
                                    self_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                else:
                                    inter_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                        data = {}
                        data['self_information'] = self_information
                        data['inter_information'] = inter_information
                        data['context_information'] = context_information
                        data['label'] = train_map[i][j + Turn][0]
                        train_data.append(data)
                        num = num + 1
                    else:
                        self_information = []
                        inter_information = []
                        context_information = []

                        for x in range(Turn):
                            for y in range(len(train_map[i][j + x])):
                                if (x % 2 == 0):
                                    self_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                else:
                                    inter_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                        for wx in range(w):
                            self_information.append(train_map[i][j + Turn][wx])
                            context_information.append(train_map[i][j + Turn][wx])
                        data = {}
                        data['self_information'] = self_information
                        data['inter_information'] = inter_information
                        data['context_information'] = context_information
                        data['label'] = train_map[i][j + Turn][0]
                        train_data.append(data)
                        num = num + 1
        train_data_ALL.append(train_data)

    print(num)

    label_list= [1,2,3,4,5]
    num = 0
    traindata_1 = []
    for i in range(len(train_data_ALL)):
        input_traindata_1_1 = []
        for j in range(len(train_data_ALL[i])):
            a = {}
            if (train_data_ALL[i][j]['label']['label'] in label_list):
                if (train_data_ALL[i][j]['label']['label'] == 5):
                    train_data_ALL[i][j]['label']['label'] = 2
                a['label_self_emotion_change'] = int(train_data_ALL[i][j]['label']['Self_Emotion_Change'])
                a['label_inter_emotion_change'] = int(train_data_ALL[i][j]['label']['Inter_Emotion_Change'])
                a['trad_data'] = Get_data(train_data_ALL[i][j],'trad_data')
                a['label_emotion'] = int(train_data_ALL[i][j]['label']['label']-1)
                #a['spec_data'] = input_traindata_x_3[i][x]
                a['transcr_data'] = Get_data(train_data_ALL[i][j],'transcr_data')
                a['id'] = Get_data(train_data_ALL[i][j],'id')
                a['section_id'] = train_data_ALL[i][j]['label']['id'][4]
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

train_org_data = Design_Turn(train_org_data_map)

train_org_data_map = Design_emotion_change(train_org_data)
Train_data = Train_data(train_org_data_map)
file = open('Train_data.pickle', 'wb')
pickle.dump(Train_data, file)