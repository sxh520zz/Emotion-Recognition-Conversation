import pickle
import os
import numpy as np
import csv
import pandas as pd


Data_dir = 'E:\ALL_Dataset'
rootdir = Data_dir + '/MELD.Raw/'

def Get_fea(dir):
    traindata = []
    num = 0
    for sess in os.listdir(dir):
        data_dir = dir + '/' + sess
        data_1 = []
        data = {}
        file = open(data_dir, 'r')
        file_content = csv.reader(file)
        for row in file_content:
            x = []
            for i in range(len(row)):
                row[i] = float(row[i])
                b = np.isinf(row[i])
                # print(b)
                if b:
                    print(row[i])
                x.append(row[i])
            row = np.array(x)
            data_1.append(row)
        data['id'] = sess[:-4]
        data_1_1 = np.array(data_1)
        data['fea_data'] = data_1_1.T
        num = num + 1
        traindata.append(data)
        print(num)
    print(len(traindata))
    return traindata

def Get_label(label_file,name):
    train_label = []
    max_dia_num = 0
    for sess in os.listdir(label_file):
        data_name = name + "_sent_emo.csv"
        if(sess == data_name):
            Data_label_file = label_file + sess
            file = open(Data_label_file,errors='ignore')
            file_content = csv.reader(file)
            for row in file_content:
                if(row[0][0] != 'S'):
                    s_data = {}
                    s_data['Utterance'] = row[1]
                    s_data['Speaker'] = row[2]
                    s_data['Emotion'] = row[3]
                    s_data['Sentiment'] = row[4]
                    s_data['Dialogue_ID'] = row[5]
                    s_data['Utterance_ID'] = row[6]
                    s_data['Season'] = row[6]
                    s_data['Episode'] = row[7]
                    s_data['StartTime'] = row[8]
                    s_data['EndTime'] = row[9]
                    if (max_dia_num < int(s_data['Dialogue_ID'])):
                        max_dia_num = int(s_data['Dialogue_ID'])
                    train_label.append(s_data)
    Train_data = [[] for x in range(max_dia_num+1)]


    for data_ind in train_label:
        Train_data[int(data_ind['Dialogue_ID'])].append(data_ind)

    train_data = []
    for i in range(len(Train_data)):
        dia_data = {}
        name_list = []
        for j in range(len(Train_data[i])):
            if(Train_data[i][j]['Speaker'] not in name_list):
                name_list.append(Train_data[i][j]['Speaker'])
        dia_data['Speaker_num'] = len(name_list)
        dia_data['Dia_length'] = len(Train_data[i])
        dia_data['Dia_data'] = Train_data[i]
        train_data.append(dia_data)
    return train_data



def Class_data(all_data):
    num = 0
    data_num = 0
    for i in range(len(all_data)):
        #if(all_data[i]['Speaker_num'] == 10 and all_data[i]['Dia_length'] >= 6):
        if (all_data[i]['Speaker_num'] == 1):
            num = num +1
            data_num = data_num + all_data[i]['Dia_length']
    print(data_num)

#all_data = Get_fea(fea_file)
train_label = Get_label(rootdir,'train')
dev_label = Get_label(rootdir,'dev')
test_label = Get_label(rootdir,'test')

Class_data(train_label)
Class_data(dev_label)
Class_data(test_label)

'''
#list_train = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
list_train = [1,2,3,4,5,6,7,8,9,10]

#train_data = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
train_data = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(All_data_class)):
    train_data[All_data_class[i]['Group']-1].append(All_data_class[i])

file = open('train_data.pickle', 'wb')
pickle.dump(train_data,file)
file.close()


for i in range(len(All_label)):
    for j in range(len(All_label[i])):
        fea = []
        for x in range(len(All_data_class[i])):
            if(str(j+1) == str(All_data_class[i][x]['time_class'])):
                fea.append(All_data_class[i][x]['fea_data'])
        All_label[i][j]['ALL_fea_data'] = fea
        print(j)
a = [0.0 for i in range(5)]
a = np.array(a)
lens = []
for i in range(len(All_label)):
    for j in range(len(All_label[i])):
        ha = []
        if(len(All_label[i][j]['ALL_fea_data']) < 5):
            for z in range(len(All_label[i][j]['ALL_fea_data'])):
                ha.append(np.array(All_label[i][j]['ALL_fea_data'][z]))
            len_zero = 5 - len(All_label[i][j]['ALL_fea_data'])
            for x in range(len_zero):
                ha.append(a)
        else:
            for z in range(len(All_label[i][j]['ALL_fea_data'])):
                if(z < 5):
                    ha.append(np.array(All_label[i][j]['ALL_fea_data'][z]))
        All_label[i][j]['ALL_fea_data'] = ha
train_data = []
test_data = []
for i in range(len(All_label)):
    if(All_label[i][0]['id'][0] == 't'):
        train_data.append(All_label[i])
    if (All_label[i][0]['id'][0] == 'd'):
        test_data.append(All_label[i])
print(len(train_data))
print(len(test_data))
print(train_data[0][:5])
print(test_data[0][:5])
file = open('train_data.pickle', 'wb')
pickle.dump(train_data,file)
file.close()
file = open('test_data.pickle', 'wb')
pickle.dump(test_data,file)
file.close()
'''