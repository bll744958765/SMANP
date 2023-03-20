#coding=gbk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#Split training set and test set Randomly

data_frame = pd.read_csv('./data.csv')
all_data = np.array(data_frame)

train_data, valid_data = train_test_split(all_data, test_size=0.7)
train_data=pd.DataFrame(train_data)
valid_data=pd.DataFrame(valid_data)
print(len(train_data),len(valid_data))
train_data.to_csv('./train.csv', index=False)
valid_data.to_csv('./valid.csv', index=False)
