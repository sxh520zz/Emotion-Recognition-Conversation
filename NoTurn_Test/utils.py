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
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

class subDataset(Dataset.Dataset):
    def __init__(self,Data_1,Data_2,Label,Label_1):
        self.Data_1 = Data_1
        self.Data_2 = Data_2
        self.Label = Label
        self.Label_1 = Label_1
    def __len__(self):
        return len(self.Data_1)
    def __getitem__(self, item):
        data_1 = self.Data_1[item]
        data_2 = self.Data_2[item]

        data_1_1 = data_1[0]
        data_1_2 = data_1[1]
        data_1_3 = data_1[2]

        data_2_1 = data_2[0]
        data_2_2 = data_2[1]
        data_2_3 = data_2[2]

        data_1_1 = torch.Tensor(np.array((data_1_1)))
        data_1_2 = torch.Tensor(np.array((data_1_2)))
        data_1_3 = torch.Tensor(np.array((data_1_3)))

        data_2_1 = torch.Tensor(np.array((data_2_1)))
        data_2_2 = torch.Tensor(np.array((data_2_2)))
        data_2_3 = torch.Tensor(np.array((data_2_3)))

        label = torch.Tensor(self.Label[item])
        label_1 = torch.Tensor(self.Label_1[item])

        return data_1_1,data_1_2,data_1_3,data_2_1,data_2_2,data_2_3,label,label_1


def Padding(args,data):
    a = [0.0 for i in range(args.utt_insize)]
    a = np.array(a)
    input_data_spec_CNN = []
    for i in range(len(data)):
        input_data_spec_CNN_1 = []
        for j in range(len(data[i])):
            ha = []
            if(len(data[i][j]) < 300):
                for z in range(len(data[i][j])):
                    ha.append(np.array(data[i][j][z]))
                len_zero = 300 - len(data[i][j])
                for x in range(len_zero):
                    ha.append(np.array(a))
            if(len(data[i][j]) >= 300):
                for z in range(len(data[i][j])):
                    if(z < 300):
                        ha.append(np.array(data[i][j][z]))
            ha = np.array(ha)
            input_data_spec_CNN_1.append(ha)
        input_data_spec_CNN.append(input_data_spec_CNN_1)
    return input_data_spec_CNN

def Feature(args,data):
    input_train_data_trad = []
    for i in range(len(data)):
        input_train_data_trad.append(np.array(data[i]['trad_data']))

    input_train_data_tran = []
    for i in range(len(data)):
        input_train_data_tran.append(data[i]['transcr_data'])

    input_label = []
    for i in range(len(data)):
        print(data[i]['label_emotion'])
        input_label.append(data[i]['label_emotion'])

    input_label_emotionchange = []
    for i in range(len(data)):
        input_label_emotionchange.append(data[i]['label_self_emotion_change'])

    input_data_id= []
    for i in range(len(data)):
        input_data_id.append(data[i]['id'][0][0:-5])
    input_orgin_label = []
    for i in range(len(data)):
        input_orgin_label.append(data[i]['label_emotion'])


    return input_train_data_trad,input_train_data_tran,input_label,input_label_emotionchange,input_data_id,input_orgin_label

def Get_data(data,train,test,args):
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])
    for i in range(len(test)):
        test_data.extend(data[test[i]])

    i = 0
    org_len = len(test_data)
    if (len(test_data) % args.batch_size != 0):
        w = args.batch_size - len(test_data) % args.batch_size
        while (i < w):
            test_data.append(test_data[0])
            i = i + 1

    print(len(train_data))
    print(len(test_data))

    input_train_data_trad,input_train_data_tran,input_train_label,input_train_label_emotionchange,_,_ = Feature(args,train_data)
    input_test_data_trad,input_test_data_tran,input_test_label,input_test_label_emotionchange,input_test_data_id,input_test_label_org = Feature(args,test_data)

    label = np.array(input_train_label).reshape(-1, 1)
    label_1 = np.array(input_train_label_emotionchange).reshape(-1, 1)

    label_test = np.array(input_test_label).reshape(-1,1)
    label_test_1 = np.array(input_test_label_emotionchange).reshape(-1,1)

    class_num_0 = 0
    class_num_1 = 0
    for i in range(len(input_train_label)):
        if(input_train_label_emotionchange[i] == 0):
            class_num_0 = class_num_0 + 1
        if(input_train_label_emotionchange[i] == 1):
            class_num_1 = class_num_1 + 1

    train_dataset = subDataset(input_train_data_trad,input_train_data_tran,label,label_1)
    test_dataset = subDataset(input_test_data_trad,input_test_data_tran,label_test,label_test_1)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return train_loader, test_loader, 1/class_num_0, 1/class_num_1, input_test_data_id[:org_len], input_test_label[:org_len], org_len