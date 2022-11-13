# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle

# reload a file to a variable
with open('train_data_map.pickle', 'rb') as file:
    train_data= pickle.load(file)
with open('test_data_map.pickle', 'rb') as file:
    test_data = pickle.load(file)

Turn = 4

def Check_data(data):
    train_data = []
    for i in range(len(data)):
        train_data_1 = []
        for j in range(len(data[i])):
            if(len(data[i][j]) == 12):
                train_data_1.append(data[i][j])
        train_data.append(train_data_1)
    return train_data

def emo_change(x):
    if x == 'neutral':
        x = 0
    if x == 'joy':
        x = 1
    if x == 'anger':
        x = 2
    if x == 'sadness':
        x = 3
    return x

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
    '''
    if(len(Out_data_0) == 0):
        print(len(Out_data_0))
    '''

    Out_data = [Out_data_0,Out_data_1,Out_data_2]
    return Out_data

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


    return Turn_Data,Turn_Data_name,Turn_Data_name_list

def Design_emotion_change(train_org_data_map,train_org_name_map,train_org_name_list_map):
    no_change = 0
    change = 0
    for i in range(len(train_org_data_map)):
        emo_dic = [-1 for _ in range(len(train_org_name_list_map[i]))]
        #对话的第一句
        speaker = train_org_data_map[i][0][-1]['Speaker']
        intorlocutor = train_org_data_map[i][1][-1]['Speaker']
        emo_dic[train_org_name_list_map[i].index(speaker)] = train_org_data_map[i][0][-1]['Emotion']
        emo_dic[train_org_name_list_map[i].index(intorlocutor)] = train_org_data_map[i][1][-1]['Emotion']
        #对话开始
        for j in range(2,len(train_org_data_map[i])):
            for y in range(len(train_org_data_map[i][j])):
                speaker = train_org_data_map[i][j][y]['Speaker']
                intorlocutor_now = intorlocutor
                #如果是新的说话人
                if(emo_dic[train_org_name_list_map[i].index(speaker)] == -1 ):
                    train_org_data_map[i][j][y]['Self_Emotion_Change'] = 0
                    if(train_org_data_map[i][j][y]['Emotion'] == emo_dic[train_org_name_list_map[i].index(intorlocutor_now)]):
                        train_org_data_map[i][j][y]['Inter_Emotion_Change'] = 0
                    else:
                        train_org_data_map[i][j][y]['Inter_Emotion_Change'] = 1
                    intorlocutor = speaker
                    emo_dic[train_org_name_list_map[i].index(speaker)] = train_org_data_map[i][j][y]['Emotion']
                else:
                    if(train_org_data_map[i][j][y]['Emotion'] == emo_dic[train_org_name_list_map[i].index(speaker)]):
                        train_org_data_map[i][j][y]['Self_Emotion_Change'] = 0
                    else:
                        train_org_data_map[i][j][y]['Self_Emotion_Change'] = 1

                    if(train_org_data_map[i][j][y]['Emotion'] == emo_dic[train_org_name_list_map[i].index(intorlocutor_now)]):
                        train_org_data_map[i][j][y]['Inter_Emotion_Change'] = 0
                    else:
                        train_org_data_map[i][j][y]['Inter_Emotion_Change'] = 1
                    intorlocutor = speaker
                    emo_dic[train_org_name_list_map[i].index(speaker)] = train_org_data_map[i][j][y]['Emotion']

    num = 0
    train_data = []
    train_name = []
    for i in range(len(train_org_data_map)):
        train_1 = []
        train_1_name = []
        for j in range(len(train_org_data_map[i])):
            train_2 = []
            train_2_name = []
            for y in range(len(train_org_data_map[i][j])):
                if(len(train_org_data_map[i][j][y]) == 14):
                    train_2.append(train_org_data_map[i][j][y])
                    train_2_name.append(train_org_name_map[i][j][y])
                    num = num +1
            if(train_2 != [] and train_2_name != []):
                train_1.append(train_2)
                train_1_name.append(train_2_name)
        train_data.append(train_1)
        train_name.append(train_1_name)
    print(num)

    num = 0
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            for x in range(len(train_data[i][j])):
                num = num +1
    print(num)
    #print(no_change)
    #print(change)
    #print(no_change+change)
    return train_data,train_name

def Train_data(train_map,train_name):
    num = 0
    '''
    for i in range(len(train_name)):
        for j in range(len(train_name[i])):
            print(train_name[i][j])
        break
    '''
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
                        data['label'] = train_map[i][j + Turn][0]
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
            if (train_data_ALL[i][j]['label']['Emotion'] in label_list):
                a['label_self_emotion_change'] = int(train_data_ALL[i][j]['label']['Self_Emotion_Change'])
                a['label_inter_emotion_change'] = int(train_data_ALL[i][j]['label']['Inter_Emotion_Change'])
                a['trad_data'] = Get_data(train_data_ALL[i][j],'wav_fea')
                a['label_emotion'] = int(train_data_ALL[i][j]['label']['Emotion'])
                #a['spec_data'] = input_traindata_x_3[i][x]
                a['transcr_data'] = Get_data(train_data_ALL[i][j],'Utterance')
                a['id'] = Get_data(train_data_ALL[i][j],'Id')
                traindata_1.append(a)
                num = num + 1
    print(num)
    return traindata_1


train_data = Check_data(train_data)
test_data = Check_data(test_data)

train_org_data, train_org_name,train_org_name_list = Design_Turn(train_data)
test_org_data,test_org_name,test_org_name_list = Design_Turn(test_data)

train_org_data_map,train_name = Design_emotion_change(train_org_data, train_org_name,train_org_name_list)
test_org_data_map,test_name = Design_emotion_change(test_org_data, test_org_name,test_org_name_list)

train_data_fin = Train_data(train_org_data_map,train_name)
test_data_fin = Train_data(test_org_data_map,test_name)

Train_data = []
Train_data.append(train_data_fin)
Train_data.append(test_data_fin)

file = open('Train_data.pickle', 'wb')
pickle.dump(Train_data, file)

