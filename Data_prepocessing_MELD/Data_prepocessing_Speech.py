import pickle
import os
import numpy as np
import csv
import pandas as pd


Data_dir = '/home/shixiaohan-toda/Documents/DataBase/journal_Data'
rootdir = Data_dir + '/MELD.Raw/'
rootdir_1 = Data_dir + '/MELD.Extra/'

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

def Get_fea(dir,name):
    data_dir_org = dir + name + '_data'
    traindata = []
    num = 0
    for sess in os.listdir(data_dir_org):
        data_dir = data_dir_org + '/' + sess
        data_1 = []
        data = {}
        file = open(data_dir, 'r')
        file_content = csv.reader(file)
        for row in file_content:
            if(row[0] != 'file'):
                x = []
                for i in range(3,len(row)):
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
    return traindata
def Get_label(label_file,name):
    train_label = []
    max_dia_num = 0
    emo_label = [0,1,2,3]
    for sess in os.listdir(label_file):
        data_name = name + "_sent_emo.csv"
        if(sess == data_name):
            Data_label_file = label_file + sess
            file = open(Data_label_file,errors='ignore')
            file_content = csv.reader(file)
            for row in file_content:
                if(row[0][0] != 'S'):
                    s_data = {}
                    s_data['Id'] = 'dia'+ row[5] + '_utt' + row[6]
                    s_data['Utterance'] = row[1]
                    s_data['Speaker'] = row[2]
                    s_data['Emotion'] = emo_change(row[3])
                    s_data['Sentiment'] = row[4]
                    s_data['Dialogue_ID'] = row[5]
                    s_data['Utterance_ID'] = row[6]
                    s_data['Season'] = row[6]
                    s_data['Episode'] = row[7]
                    s_data['StartTime'] = row[8]
                    s_data['EndTime'] = row[9]
                    if (max_dia_num < int(s_data['Dialogue_ID'])):
                        max_dia_num = int(s_data['Dialogue_ID'])
                    if (s_data['Emotion'] in emo_label):
                        train_label.append(s_data)
    return train_label,max_dia_num
def combine_wav_text(wav_data,text_data):
    for i in range(len(wav_data)):
        for j in range(len(text_data)):
            if(wav_data[i]['id'] == text_data[j]['Id']):
                text_data[j]['wav_fea'] = wav_data[i]['fea_data']
    return  text_data
                
def Class_data(all_data,max_dia_num):
    Train_data = [[] for x in range(max_dia_num+1)]
    for data_ind in all_data:
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

    CAN_USE = []
    for i in range(len(train_data)):
        if(train_data[i]['Speaker_num'] == 2 and train_data[i]['Dia_length'] >= 7):
            CAN_USE.append(train_data[i]['Dia_data'])

    return  CAN_USE


train_pre_data = Get_fea(rootdir_1,'train')
test_pre_data = Get_fea(rootdir_1,'test')

train_label,max_dia_num_train = Get_label(rootdir,'train')
test_label,max_dia_num_test = Get_label(rootdir,'test')

train_data = combine_wav_text(train_pre_data,train_label)
test_data = combine_wav_text(test_pre_data,test_label)

train_data_org = Class_data(train_data,max_dia_num_train)
test_data_org = Class_data(test_data,max_dia_num_test)

file = open('train_data_org.pickle', 'wb')
pickle.dump(train_data_org, file)
file.close()
file = open('test_data_org.pickle', 'wb')
pickle.dump(test_data_org, file)
file.close()


