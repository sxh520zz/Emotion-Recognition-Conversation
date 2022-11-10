# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""
import gensim
from bert_serving.client import BertClient
import pickle

# reload a file to a variable
with open('train_data_org.pickle', 'rb') as file:
    train_org_data = pickle.load(file)
with open('test_data_org.pickle', 'rb') as file:
    test_org_data= pickle.load(file)
bc = BertClient()

def Get_Bert(train_org_data_map):
    for i in range(len(train_org_data_map)):
        for j in range(len(train_org_data_map[i])):
            a = train_org_data_map[i][j]['Utterance']
            line = gensim.utils.simple_preprocess(a)
            b = ' '.join(line)
            if b:
                train_org_data_map[i][j]['Utterance'] = b
            else:
                train_org_data_map[i][j]['Utterance']
    x = 0
    for i in range(len(train_org_data_map)):
        for j in range(len(train_org_data_map[i])):
            z = []
            a = train_org_data_map[i][j]['Utterance']
            z.append(a)
            train_org_data_map[i][j]['Utterance'] = bc.encode(z)
            x = x + 1
            if (x % 100 == 0):
                print(x)
    print(x)
    return train_org_data_map

train_orga_map = Get_Bert(train_org_data)
test_orga_map = Get_Bert(test_org_data)

print(train_orga_map[0][0])
file = open('train_data_map.pickle', 'wb')
pickle.dump(train_orga_map, file)
file = open('test_data_map.pickle', 'wb')
pickle.dump(test_orga_map, file)

