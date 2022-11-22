# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle

# reload a file to a variable
with open('../MELD_Feature/train_data_map_for-Mulit-speaker.pickle', 'rb') as file:
    train_data= pickle.load(file)
with open('../MELD_Feature/test_data_map_for-Mulit-speaker.pickle', 'rb') as file:
    test_data = pickle.load(file)

Turn = 6

def Check_data(data):
    num = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            num = num +1
    print(num)

def emo_change(x):
    if x == 'neutral':
        x = 0
    if x == 'joy':
        x = 1
    if x == 'anger':
        x = 2
    if x == 'sadness':
        x = 3
    '''
    if x == 'sadness':
        x = 4
    if x == 'sadness':
        x = 5
    if x == 'sadness':
        x = 6
    '''

    return x

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
    '''
    if(len(Out_data_0) == 0):
        print(len(Out_data_0))
    '''

    Out_data = [Out_data_0,Out_data_1,Out_data_2]
    return Out_data

def remove_nested_list(listt):
    while [] in listt:  # 判断是否有空值在列表中
        listt.remove([])  # 如果有就直接通过remove删除
    return listt

def Design_Turn(train_org_data):
    Turn_Data = []
    Turn_Data_name = []
    Turn_Data_name_list = []
    for i in range(len(train_org_data)):
        data_turn_level = [[] for _ in range(500)]
        data_turn_name = [[] for _ in range(500)]
        data_turn_name_list = []
        Turn_utt = 0
        for x in range(len(train_org_data[i])):
            #标签
            train_org_data[i][x]['Emotion'] = emo_change(train_org_data[i][x]['Emotion'])

            if(train_org_data[i][x]['Speaker'] not in data_turn_name_list):
                data_turn_name_list.append(train_org_data[i][x]['Speaker'])

            if (x == 0):
                data_turn_level[Turn_utt].append(train_org_data[i][x])
                data_turn_name[Turn_utt].append(train_org_data[i][x]['Speaker'])
            elif(train_org_data[i][x]['Speaker'] == data_turn_level[Turn_utt][0]['Speaker']):
                data_turn_level[Turn_utt].append(train_org_data[i][x])
                data_turn_name[Turn_utt].append(train_org_data[i][x]['Speaker'])
            else:
                Turn_utt = Turn_utt + 1
                data_turn_level[Turn_utt].append(train_org_data[i][x])

                data_turn_name[Turn_utt].append(train_org_data[i][x]['Speaker'])
        Turn_Dia = remove_nested_list(data_turn_level)
        Turn_Dia_name = remove_nested_list(data_turn_name)

        Turn_Data.append(Turn_Dia)
        Turn_Data_name.append(Turn_Dia_name)
        Turn_Data_name_list.append(data_turn_name_list)

    Check_data(Turn_Data_name)
    Check_data(Turn_Data)
    return Turn_Data,Turn_Data_name,Turn_Data_name_list

def Train_data(train_map,train_name):
    num = 0
    train_data_ALL_1 = []
    for i in range(len(train_map)):
        train_data = []
        for j in range(len(train_map[i]) - Turn):
            if(len(train_map[i][j + Turn]) == 1):
                self_information = []
                inter_information = []
                context_information = []
                for x in range(Turn):
                    for y in range(len(train_map[i][j + x])):
                        if (train_name[i][j + x][y] == train_name[i][j + Turn][-1]):
                            self_information.append(train_map[i][j + x][y])
                            context_information.append(train_map[i][j + x][y])
                        elif(train_name[i][j + x][y] == train_name[i][j + Turn-1][-1]):
                            inter_information.append(train_map[i][j + x][y])
                            context_information.append(train_map[i][j + x][y])
                        else:
                            context_information.append(train_map[i][j + x][y])
                data = {}
                data['self_information'] = self_information
                data['inter_information'] = inter_information
                data['context_information'] = context_information
                data['label'] = train_map[i][j + Turn][0]['Emotion']
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
                                if (train_name[i][j + x][y] == train_name[i][j + Turn][-1]):
                                    self_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                elif (train_name[i][j + x][y] == train_name[i][j + Turn - 1][-1]):
                                    inter_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                else:
                                    context_information.append(train_map[i][j + x][y])
                        data = {}
                        data['self_information'] = self_information
                        data['inter_information'] = inter_information
                        data['context_information'] = context_information
                        data['label'] = train_map[i][j + Turn][0]['Emotion']
                        train_data.append(data)
                        num = num + 1
                    else:
                        self_information = []
                        inter_information = []
                        context_information = []
                        for x in range(Turn):
                            for y in range(len(train_map[i][j + x])):
                                if (train_name[i][j + x][y] == train_name[i][j + Turn][-1]):
                                    self_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                elif (train_name[i][j + x][y] == train_name[i][j + Turn - 1][-1]):
                                    inter_information.append(train_map[i][j + x][y])
                                    context_information.append(train_map[i][j + x][y])
                                else:
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
        train_data_ALL_1.append(train_data)

    num = 0
    train_data_ALL = []
    for i in range(len(train_data_ALL_1)):
        xxx = []
        for j in range(len(train_data_ALL_1[i])):
            if(len(train_data_ALL_1[i][j]['inter_information']) != 0 and len(train_data_ALL_1[i][j]['self_information']) != 0):
                xxx.append(train_data_ALL_1[i][j])
                num = num +1
        train_data_ALL.append(xxx)
    print(num)

    label_list= [0,1,2,3]
    num = 0
    traindata_1 = []
    for i in range(len(train_data_ALL)):
        for j in range(len(train_data_ALL[i])):
            a = {}
            if (train_data_ALL[i][j]['label']in label_list):
                a['trad_data'] = Get_data(train_data_ALL[i][j],'wav_fea')
                a['label_emotion'] = int(train_data_ALL[i][j]['label'])
                #a['spec_data'] = input_traindata_x_3[i][x]
                a['transcr_data'] = Get_data(train_data_ALL[i][j],'Utterance')
                a['id'] = Get_data(train_data_ALL[i][j],'Id')
                traindata_1.append(a)
                num = num + 1
    print('可用的数据为 ', num)
    return traindata_1




train_org_data, train_org_name,train_org_name_list = Design_Turn(train_data)
test_org_data,test_org_name,test_org_name_list = Design_Turn(test_data)

train_data_fin = Train_data(train_org_data, train_org_name)
test_data_fin = Train_data(test_org_data, test_org_name)

Train_data = []
Train_data.append(train_data_fin)
Train_data.append(test_data_fin)
file = open('../MELD_Feature/Train_data_Multi_Speaker.pickle', 'wb')
pickle.dump(Train_data, file)



