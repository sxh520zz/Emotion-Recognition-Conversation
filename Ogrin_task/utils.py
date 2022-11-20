import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.utils.data.dataset as Dataset
from sklearn import preprocessing
import torch.optim as optim
from torch.autograd import Variable
from models import Utterance_net
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold


class subDataset(Dataset.Dataset):
    def __init__(self,Data_1,Data_2,Label):
        self.Data_1 = Data_1
        self.Data_2 = Data_2
        self.Label = Label
    def __len__(self):
        return len(self.Data_1)
    def __getitem__(self, item):
        data_1 = self.Data_1[item]
        data_1_1 = self.Data_2[item]
        data_2 = []
        data_2_1 = []
        data_3 = []
        data_3_1 = []
        for i in range(len(data_1)):
            if(i % 2 == 0):
                data_2.append(data_1[i])
            else:
                data_3.append(data_1[i])

        for i in range(len(data_1_1)):
            if(i % 2 == 0):
                data_2_1.append(data_1_1[i])
            else:
                data_3_1.append(data_1_1[i])

        data_1 = torch.Tensor(data_1)
        data_2 = torch.Tensor(data_2)
        data_3 = torch.Tensor(data_3)

        data_1_1 = torch.Tensor(data_1_1)
        data_2_1 = torch.Tensor(data_2_1)
        data_3_1 = torch.Tensor(data_3_1)

        label = torch.Tensor(self.Label[item])

        return data_1,data_2,data_3,data_1_1,data_2_1,data_3_1,label

def Padding(args,data):
    a = [0.0 for i in range(args.utt_insize)]
    a = np.array(a)
    input_data_spec_CNN = []
    for i in range(len(data)):
        ha = []
        if(len(data[i]) < 300):
            for z in range(len(data[i])):
                ha.append(np.array(data[i][z]))
            len_zero = 300 - len(data[i])
            for x in range(len_zero):
                ha.append(np.array(a))
        if(len(data[i]) >= 300):
            for z in range(len(data[i])):
                if(z < 300):
                    ha.append(np.array(data[i][z]))
        ha = np.array(ha)
        input_data_spec_CNN.append(ha)
    return input_data_spec_CNN

def Feature(args,data):
    input_train_data_trad = []
    for i in range(len(data)):
        input_train_data_trad.append(np.array(data[i]['trad_data']))

    input_train_data_tran = []
    for i in range(len(data)):
        input_train_data_tran.append(data[i]['transcr_data'])
    '''
    for i in range(len(input_train_data_tran)):
        for j in range(len(input_train_data_tran[i])):
            input_train_data_tran[i][j] = Padding(args,input_train_data_tran[i][j])
    '''

    input_label = []
    for i in range(len(data)):
        input_label.append(data[i]['label_emotion'])

    input_label_self_Emotion_Change = []
    for i in range(len(data)):
        input_label_self_Emotion_Change.append(data[i]['label_emotion_change_self'])

    input_label_Inter_Emotion_Change = []
    for i in range(len(data)):
        input_label_Inter_Emotion_Change.append(data[i]['label_emotion_change_into'])

    input_data_id= []
    for i in range(len(data)):
        input_data_id.append(data[i]['id'][0][0:-5])
    input_orgin_label = []
    for i in range(len(data)):
        input_orgin_label.append(data[i]['label_emotion'])

    return input_train_data_trad,input_train_data_tran,input_label,input_data_id,input_orgin_label

def Get_data(data,train,test,args):
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])
    for i in range(len(test)):
        test_data.extend(data[test[i]])

    print(len(train_data))
    print(len(test_data))

    org_len = len(test_data)
    if (len(test_data) % args.batch_size != 0):
        w = args.batch_size - len(test_data) % args.batch_size
        while (i < w):
            test_data.append(test_data[0])
            i = i + 1
    '''
    add_data = []
    for i in range(len(train_data)):
        if(train_data[i]['label_emotion_change'] == 0):
            add_data.append(data[i])
    for i in range(len(add_data)):
        train_data.extend(add_data[i])
    '''



    input_train_data_trad,input_train_data_tran,input_train_label,_,_ = Feature(args,train_data)
    input_test_data_trad,input_test_data_tran,input_test_label,input_test_data_id,input_test_label_org = Feature(args,test_data)

    label = np.array(input_train_label).reshape(-1, 1)
    label_test = np.array(input_test_label).reshape(-1,1)


    train_dataset = subDataset(input_train_data_trad,input_train_data_tran,label)
    test_dataset = subDataset(input_test_data_trad,input_test_data_tran,label_test)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return train_loader, test_loader, input_test_data_id[:org_len], input_test_label[:org_len], org_len