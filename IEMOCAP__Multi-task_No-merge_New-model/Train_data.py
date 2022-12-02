# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle
import math
import numpy as np
import numpy
# reload a file to a variable
with open('../IEM_Feature/Text_data_No_combine.pickle', 'rb') as file:
    train_org_data_map = pickle.load(file)

Turn = 6

#print(len(train_org_data_map))
#print(train_org_data_map[0][0])


def included_angle(a, b):
    a_norm = np.sqrt(np.sum(a * a))
    b_norm = np.sqrt(np.sum(b * b))
    cos_value = np.dot(a, b) / (a_norm * b_norm)
    arc_value = np.arccos(cos_value)
    angle_value = arc_value * 180 / np.pi
    return angle_value

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = -int(angle1 * 180 / math.pi)
    if angle1 < 0:
        angle1 = 360 + angle1

    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = - int(angle2 * 180 / math.pi)
    if angle2 < 0:
        angle2 = 360 + angle2

    #print(angle1, angle2)

    included_angle = angle1 - angle2

    if abs(included_angle) > 180:
        included_angle = included_angle / abs(included_angle) * (360 - abs(included_angle))
    else:
        included_angle *= -1
    return included_angle

def remove_nested_list(listt):
    while [] in listt:  # 判断是否有空值在列表中
        listt.remove([])  # 如果有就直接通过remove删除
    return listt

def Pre_did(all_data):
    #按照话轮重新整理数据库
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
# 失败案例 - From 中科大
def Design_emotion_change(train_map):

    a= []
    for i in range(len(train_map)):
        x_data = []
        for j in range(2,len(train_map[i])):
            if(len(train_map[i][j]) == 1):
                if(train_map[i][j][-1]['label'] == 0 or train_map[i][j -2][-1]['label'] == 0):
                    train_map[i][j][-1]['Self_Emotion_Change'] = 1/6
                    #x.append(train_org_data_map[i][j])
                elif(train_map[i][j][-1]['emotion_v'] *  train_map[i][j -2][-1]['emotion_v'] < 0):
                    train_map[i][j][-1]['Self_Emotion_Change'] = 0
                else:
                    data_1 = [0,0,train_map[i][j][-1]['emotion_v'],train_map[i][j][-1]['emotion_a']]
                    data_2 = [0,0,train_map[i][j -2][-1]['emotion_v'], train_map[i][j -2][-1]['emotion_a']]
                    angele = angle(data_1,data_2)
                    train_map[i][j][-1]['Self_Emotion_Change'] = max(math.cos(angele),0)
            else:
                for x in range(len(train_map[i][j])):
                    if (x == 0):
                        if (train_map[i][j][x]['label'] == 0 or train_map[i][j - 2][-1]['label'] == 0):
                            train_map[i][j][x]['Self_Emotion_Change'] = 1/6
                        elif (train_map[i][j][x]['emotion_v'] * train_map[i][j - 2][-1]['emotion_v'] < 0):
                            train_map[i][j][x]['Self_Emotion_Change'] = 0
                        else:
                            data_1 = [0, 0, train_map[i][j][x]['emotion_v'], train_map[i][j][x]['emotion_a']]
                            data_2 = [0, 0, train_map[i][j - 2][-1]['emotion_v'], train_map[i][j - 2][-1]['emotion_a']]
                            angele = angle(data_1, data_2)
                            train_map[i][j][x]['Self_Emotion_Change'] = max(math.cos(angele), 0)
                    else:
                        if (train_map[i][j][x]['label'] == 0 or train_map[i][j][x-1]['label'] == 0):
                            train_map[i][j][x]['Self_Emotion_Change'] = 1/6
                        elif(train_map[i][j][x]['emotion_v'] * train_map[i][j - 2][-1]['emotion_v'] < 0):
                            train_map[i][j][x]['Self_Emotion_Change'] = 0
                        else:
                            data_1 = [0, 0, train_map[i][j][x]['emotion_v'], train_map[i][j][x]['emotion_a']]
                            data_2 = [0, 0, train_map[i][j][x-1]['emotion_v'], train_map[i][j][x-1]['emotion_a']]
                            angele = angle(data_1, data_2)
                            train_map[i][j][x]['Self_Emotion_Change'] = max(math.cos(angele), 0)
            x_data.append(train_map[i][j])
        a.append(x_data)
    return a
'''
def calculate_distance(data_1,data_2):
    return(numpy.sqrt(numpy.square(data_1[0]-data_2[0]) + numpy.square(data_1[1]-data_2[1])))
def Design_emotion_change(train_map):
    a= []
    for i in range(len(train_map)):
        x_data = []
        for j in range(2,len(train_map[i])):
            if(len(train_map[i][j]) == 1):
                data_1 = [train_map[i][j][-1]['emotion_v'],train_map[i][j][-1]['emotion_a']]
                data_2 = [train_map[i][j -2][-1]['emotion_v'], train_map[i][j -2][-1]['emotion_a']]
                distance = calculate_distance(data_1,data_2)
                train_map[i][j][-1]['Self_Emotion_Change'] = distance
            else:
                for x in range(len(train_map[i][j])):
                    if (x == 0):
                        data_1 = [train_map[i][j][x]['emotion_v'], train_map[i][j][x]['emotion_a']]
                        data_2 = [train_map[i][j - 2][-1]['emotion_v'], train_map[i][j - 2][-1]['emotion_a']]
                        distance = calculate_distance(data_1, data_2)
                        train_map[i][j][x]['Self_Emotion_Change'] = distance
                    else:
                        data_1 = [train_map[i][j][x]['emotion_v'], train_map[i][j][x]['emotion_a']]
                        data_2 = [train_map[i][j][x-1]['emotion_v'], train_map[i][j][x-1]['emotion_a']]
                        distance = calculate_distance(data_1, data_2)
                        train_map[i][j][x]['Self_Emotion_Change'] = distance
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
                id_self = []
                id_other = []
                n_v = 0
                for x in range(Turn):
                    if(x % 2 == 0):
                        for y in range(len(train_map[i][j + x])):
                            self_information.append(train_map[i][j + x][y])
                            context_information.append(train_map[i][j + x][y])
                            id_self.append(n_v)
                            n_v = n_v + 1
                    else:
                        for y in range(len(train_map[i][j + x])):
                            inter_information.append(train_map[i][j + x][y])
                            context_information.append(train_map[i][j + x][y])
                            id_other.append(n_v)
                            n_v = n_v + 1
                data = {}
                data['self_information'] = self_information
                data['inter_information'] = inter_information
                data['context_information'] = context_information
                data['label'] = train_map[i][j + Turn][0]['label']
                data['Self_Emotion_Change'] = train_map[i][j + Turn][0]['Self_Emotion_Change']
                #data['Inter_Emotion_Change'] = train_map[i][j + Turn][0]['Inter_Emotion_Change']
                data['id'] = train_map[i][j + Turn][0]['id']
                data['Self_Turn'] = id_self
                data['Other_Turn'] = id_other
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
                        id_self = []
                        id_other = []
                        n_v = 0
                        for x in range(Turn):
                            if (x % 2 == 0):
                                for y in range(len(train_map[i][j + x])):
                                    self_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                    id_self.append(n_v)
                                    n_v = n_v + 1
                            else:
                                for y in range(len(train_map[i][j + x])):
                                    inter_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                    id_other.append(n_v)
                                    n_v = n_v + 1
                        data = {}
                        data['self_information'] = self_information
                        data['inter_information'] = inter_information
                        data['context_information'] = context_information
                        data['label'] = train_map[i][j + Turn][0]['label']
                        data['Self_Emotion_Change'] = train_map[i][j + Turn][0]['Self_Emotion_Change']
                        #data['Inter_Emotion_Change'] = train_map[i][j + Turn][0]['Inter_Emotion_Change']
                        data['id'] = train_map[i][j + Turn][0]['id']
                        data['Self_Turn'] = id_self
                        data['Other_Turn'] = id_other
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
                        id_self = []
                        id_other = []
                        n_v = 0
                        for x in range(Turn):
                            if (x % 2 == 0):
                                for y in range(len(train_map[i][j + x])):
                                    self_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                    id_self.append(n_v)
                                    n_v = n_v + 1
                            else:
                                for y in range(len(train_map[i][j + x])):
                                    inter_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                    id_other.append(n_v)
                                    n_v = n_v + 1
                        for wx in range(w):
                            self_information.append(train_map[i][j + Turn][wx])
                            context_information.append(train_map[i][j + Turn][wx])
                            id_self.append(n_v)
                            n_v = n_v + 1
                        data = {}
                        data['self_information'] = self_information
                        data['inter_information'] = inter_information
                        data['context_information'] = context_information
                        data['label'] = train_map[i][j + Turn][w]['label']
                        data['Self_Emotion_Change'] = train_map[i][j + Turn][w]['Self_Emotion_Change']
                        #data['Inter_Emotion_Change'] = train_map[i][j + Turn][w]['Inter_Emotion_Change']
                        data['id'] = train_map[i][j + Turn][w]['id']
                        data['Self_Turn'] = id_self
                        data['Other_Turn'] = id_other
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
file = open('Train_data_NEW_Model.pickle', 'wb')
pickle.dump(Train_data, file)