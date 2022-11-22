import opensmile
import pickle
import os
import numpy as np
import csv
import pandas as pd


rootdir = '/home/shixiaohan-toda/Desktop/journal/Data_prepocessing_MELD'
Data_dir = rootdir + '/test_wav'
target_dir =  rootdir + '/test_data'

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

num = 0
for sess in os.listdir(Data_dir):
    data_name = Data_dir + '/' + sess
    y = smile.process_file(data_name)
    target_name = sess[:-4] + '.csv'
    target_dir_ind = target_dir + '/' + target_name
    y.to_csv(target_dir_ind)
    num = num + 1
    print(num)

