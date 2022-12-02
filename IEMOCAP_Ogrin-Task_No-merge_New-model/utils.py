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
    def __init__(self,Data_1,Data_2,Label,Label_1,Label_2):
        self.Data_1 = Data_1
        self.Data_2 = Data_2
        self.Label = Label
        self.Label_1 = Label_1
        self.Label_2 = Label_2
    def __len__(self):
        return len(self.Data_1)
    def __getitem__(self, item):
        data_1 = self.Data_1[item]
        data_2 = self.Data_2[item]

        data_1_S = data_1[0]
        data_1_I = data_1[1]
        data_1_II = data_1[2]

        data_2_S = data_2[0]
        data_2_I = data_2[1]
        data_2_II = data_2[2]

        data_1_S = torch.Tensor(np.array(data_1_S))
        data_1_I = torch.Tensor(np.array(data_1_I))
        data_1_II = torch.Tensor(np.array(data_1_II))

        data_2_S = torch.Tensor(np.array(data_2_S))
        data_2_I = torch.Tensor(np.array(data_2_I))
        data_2_II = torch.Tensor(np.array(data_2_II))

        label = torch.Tensor(self.Label[item])
        label_1 = torch.IntTensor(np.array(self.Label_1[item]))
        label_2 = torch.IntTensor(np.array(self.Label_2[item]))

        return data_1_S,data_1_I,data_1_II,data_2_S,data_2_I,data_2_II,label,label_1,label_2

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
        data_1 = []
        self_information = []
        inter_information = []
        context_information = []
        for j in range(len(data[i]['self_information'])):
            self_information.append(np.array(data[i]['self_information'][j]['trad_data']))
        for j in range(len(data[i]['inter_information'])):
            inter_information.append(np.array(data[i]['inter_information'][j]['trad_data']))
        for j in range(len(data[i]['context_information'])):
            context_information.append(np.array(data[i]['context_information'][j]['trad_data']))
        data_1.append(self_information)
        data_1.append(inter_information)
        data_1.append(context_information)
        input_train_data_trad.append(data_1)
    input_train_data_tran = []
    for i in range(len(data)):
        data_1 = []
        self_information = []
        inter_information = []
        context_information = []
        for j in range(len(data[i]['self_information'])):
            self_information.append(np.array(data[i]['self_information'][j]['transcr_data']))
        for j in range(len(data[i]['inter_information'])):
            inter_information.append(np.array(data[i]['inter_information'][j]['transcr_data']))
        for j in range(len(data[i]['context_information'])):
            context_information.append(np.array(data[i]['context_information'][j]['transcr_data']))
        data_1.append(self_information)
        data_1.append(inter_information)
        data_1.append(context_information)
        input_train_data_tran.append(data_1)


    input_label = []
    for i in range(len(data)):
        input_label.append(data[i]['label'])

    input_label_self_Emotion_Change = []
    for i in range(len(data)):
        input_label_self_Emotion_Change.append(data[i]['Self_Turn'])

    input_label_Inter_Emotion_Change = []
    for i in range(len(data)):
        input_label_Inter_Emotion_Change.append(data[i]['Other_Turn'])

    input_data_id= []
    for i in range(len(data)):
        input_data_id.append(data[i]['id'])
    input_orgin_label = []
    for i in range(len(data)):
        input_orgin_label.append(data[i]['label'])

    return input_train_data_trad,input_train_data_tran,input_label,input_label_self_Emotion_Change,input_label_Inter_Emotion_Change,input_data_id,input_orgin_label

def Get_data(data,train,test,args):
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])
    for i in range(len(test)):
        test_data.extend(data[test[i]])

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


    print(len(train_data))
    print(len(test_data))
    input_train_data_trad,input_train_data_tran,input_train_label,input_train_label_1,input_train_label_2,_,_ = Feature(args,train_data)
    input_test_data_trad,input_test_data_tran,input_test_label,input_test_label_1,input_test_label_2,input_test_data_id,input_test_label_org = Feature(args,test_data)

    label = np.array(input_train_label).reshape(-1, 1)
    label_test = np.array(input_test_label).reshape(-1,1)

    train_dataset = subDataset(input_train_data_trad,input_train_data_tran,label,input_train_label_1,input_train_label_2)
    test_dataset = subDataset(input_test_data_trad,input_test_data_tran,label_test,input_test_label_1,input_test_label_2)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return train_loader, test_loader, input_test_data_id[:org_len], input_test_label[:org_len], org_len