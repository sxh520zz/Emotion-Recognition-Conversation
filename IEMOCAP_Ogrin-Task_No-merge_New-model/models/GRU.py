import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 04:00:26 2019

@author: SHI Xiaohan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Utterance_net_Input(nn.Module):
    def __init__(self, input_size, args):
        super(Utterance_net_Input, self).__init__()
        self.input_size = input_size
        self.hidden_dim = args.hidden_layer
        self.num_layers = args.dia_layers
        #  dropout
        self.dropout = nn.Dropout(args.dropout)
        # gru
        self.bigru = nn.GRU(input_size, self.hidden_dim,
                            batch_first=True, num_layers=self.num_layers, bidirectional=True)
    def forward(self, indata):
        embed = self.dropout(indata)
        # gru
        gru_out, gru_hid = self.bigru(embed)
        gru_out_con = torch.transpose(gru_out, 1, 2)
        gru_out_con = F.tanh(gru_out_con)
        gru_out_con = F.max_pool1d(gru_out_con, gru_out_con.size(2)).squeeze(2)
        gru_out_con = F.tanh(gru_out_con)
        return gru_out_con, gru_out

class Utterance_net(nn.Module):
    def __init__(self, input_size, args):
        super(Utterance_net, self).__init__()
        self.input_size = input_size
        self.hidden_dim = args.hidden_layer
        self.num_layers = args.dia_layers
        #  dropout
        self.dropout = nn.Dropout(args.dropout)
        # gru
        self.bigru = nn.GRU(input_size, self.hidden_dim,
                            batch_first=True, num_layers=self.num_layers, bidirectional=True)
    def forward(self, indata):
        embed = self.dropout(indata)
        # gru
        gru_out, gru_hid = self.bigru(embed)
        gru_hid = torch.transpose(gru_hid, 1, 2)
        gru_out = torch.transpose(gru_out, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = F.tanh(gru_out)
        return gru_out, gru_hid

class Dialogue_net(nn.Module):
    def __init__(self, input_size, args):
        super(Dialogue_net, self).__init__()
        self.hidden_dim = args.hidden_layer
        self.num_layers = args.dia_layers
        self.out_class = args.out_class
        #  dropout
        self.dropout = nn.Dropout(args.dropout)
        # gru
        self.bigru = nn.GRU(input_size, self.hidden_dim,
                            batch_first=True, num_layers=self.num_layers, bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.out_class)
    def forward(self, input):
        embed = self.dropout(input)
        # gru
        gru_out, gru_hid = self.bigru(embed)
        #gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = F.tanh(gru_out)
        y = self.hidden2label(gru_out)
        return y

class Output_net(nn.Module):
    def __init__(self,input_size,args):
        super(Output_net, self).__init__()
        # linear
        self.hidden_dim = args.hidden_layer
        self.out_class = args.out_class
        self.input2hidden = nn.Linear(input_size, self.hidden_dim * 2)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.out_class)

    def forward(self, input):
        # linear
        x = self.input2hidden(input)
        y = self.hidden2label(x)
        return y

class Output_net_1(nn.Module):
    def __init__(self,input_size,args):
        super(Output_net_1, self).__init__()
        # linear
        self.hidden_dim = args.hidden_layer
        self.out_class = args.out_class_1
        self.input2hidden = nn.Linear(input_size, self.hidden_dim * 2)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.out_class)

    def forward(self, input):
        # linear
        x = self.input2hidden(input)
        y = self.hidden2label(x)
        return y

class Output_net_2(nn.Module):
    def __init__(self,input_size,args):
        super(Output_net_2, self).__init__()
        # linear
        self.hidden_dim = args.hidden_layer
        self.out_class = args.out_class_1
        self.input2hidden = nn.Linear(input_size, self.hidden_dim * 2)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.out_class)

    def forward(self, input):
        # linear
        x = self.input2hidden(input)
        y = self.hidden2label(x)
        return y

