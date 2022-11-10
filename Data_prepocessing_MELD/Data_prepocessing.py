# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""
import gensim
from bert_serving.client import BertClient
import pickle

# reload a file to a variable
with open('train_data_map.pickle', 'rb') as file:
    train_data= pickle.load(file)
with open('test_data_map.pickle', 'rb') as file:
    test_data = pickle.load(file)

step = 6

def Design_emotion_change(train_org_data_map):
    no_change = 0
    change = 0
    a= []
    for i in range(len(train_org_data_map)):
        x = []
        for j in range(2,len(train_org_data_map[i])):
            if(train_org_data_map[i][j]['Emotion'] == train_org_data_map[i][j -2]['Emotion']):
                train_org_data_map[i][j]['Self_Emotion_Change'] = 0
                no_change = no_change+1
                #x.append(train_org_data_map[i][j])
            else:
                train_org_data_map[i][j]['Self_Emotion_Change'] = 1
                change = change + 1
                #x.append(train_org_data_map[i][j])
            if(train_org_data_map[i][j]['Emotion'] == train_org_data_map[i][j -1]['Emotion']):
                train_org_data_map[i][j]['Inter_Emotion_Change'] = 0
                no_change = no_change+1
                x.append(train_org_data_map[i][j])
            else:
                train_org_data_map[i][j]['Inter_Emotion_Change'] = 1
                change = change + 1
                x.append(train_org_data_map[i][j])
        a.append(x)

    #print(no_change)
    #print(change)
    #print(no_change+change)
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
                b = train_map[i][x + y]['Utterance']
                c = train_map[i][x + y]['wav_fea']
                d = train_map[i][x + y]['Id']
                input_train_tran_1.append(b)
                input_train_trad_1.append(c)
                input_train_name_1.append(d)
            input_trainlabel_1.append(train_map[i][x + step]['Self_Emotion_Change'])
            input_trainlabel_2.append(train_map[i][x + step]['Emotion'])
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

    num = 0
    traindata_1 = []
    for i in range(len(input_traindata_z)):
        input_traindata_1_1 = []
        for x in range(len(input_traindata_z[i])):
            a = {}
            a['label_emotion_change'] = int(input_traindata_x_2[i][x])
            a['label_emotion'] = int(input_traindata_x[i][x])
            a['trad_data'] = input_traindata_y[i][x]
            a['transcr_data'] = input_traindata_y_1[i][x]
            a['id'] = input_traindata_z[i][x]
            input_traindata_1_1.append(a)
            num = num + 1
        traindata_1.append(input_traindata_1_1)
    print(num)

    return traindata_1

train_org_data_map = Design_emotion_change(train_data)
test_org_data_map = Design_emotion_change(test_data)


Traindata = Train_data(train_org_data_map)
Testdata = Train_data(test_org_data_map)



Train_data = []
Train_data.append(Traindata)
Train_data.append(Testdata)

file = open('Train_data.pickle', 'wb')
pickle.dump(Train_data, file)