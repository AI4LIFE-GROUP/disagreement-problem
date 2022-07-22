import numpy as np
import pandas as pd
from IPython.display import display

import argparse



#parse arguments
parser = argparse.ArgumentParser(description='process data')
parser.add_argument('--dataset_name', type = str, choices=['compas', 'german'], help='dataset name')
args = parser.parse_args()
dataset_name = args.dataset_name



##### load data

f = open(f'{dataset_name}/data/notes-data.txt', 'w')
print('**********LOAD DATA**********\n', file=f)

#load data
path_data_train = f'{dataset_name}/data/{dataset_name}-train.csv'
path_data_test = f'{dataset_name}/data/{dataset_name}-test.csv'

data_train = pd.read_csv(path_data_train)  
data_test = pd.read_csv(path_data_test)  

#quick look
print('train set, shape:', data_train.shape, file=f)
display(data_train.head())
print('test set, shape:', data_test.shape, file=f)
display(data_test.head())



##### normalize data

print('\n**********NORMALIZE DATA**********\n', file=f)

#function to normalize data to [a, b] range
def normalize(X_train, X_test, a, b):

    #get min and max parameters from training set
    train_mins = X_train.min(axis=0)
    train_maxs = X_train.max(axis=0)

    #normalize X_train and X_test to range [a, b]
    X_train_new = (b-a)*((X_train-train_mins)/(train_maxs-train_mins)) + a
    X_test_new = (b-a)*((X_test-train_mins)/(train_maxs-train_mins)) + a
    
    return X_train_new, X_test_new

#check feature ranges, after normalization
print('----- Before normalization -----', file=f)
print('train set:', file=f)
print(data_train.describe(include='all'), file=f)
print('test set:', file=f)
print(data_test.describe(include='all'), file=f)

#normalize data
data_train_norm, data_test_norm = normalize(data_train, data_test, a=0, b=1)

#check feature ranges, after normalization
print('----- After normalization -----', file=f)
print('train set:', file=f)
print(data_train_norm.describe(include='all'), file=f)
print('test set:', file=f)
print(data_test_norm.describe(include='all'), file=f)

def check_feature_ranges(data_bf_norm, data_after_norm):
    #calculate mins and maxs, before and after normalization
    min_before = data_bf_norm.min(axis=0)
    max_before = data_bf_norm.max(axis=0)
    min_after = data_after_norm.min(axis=0)
    max_after = data_after_norm.max(axis=0)
    
    #create df
    feature_ranges = pd.DataFrame(min_before, columns=['min_before_norm'])
    feature_ranges['min_after_norm'] = min_after
    feature_ranges['max_before_norm'] = max_before
    feature_ranges['max_after_norm'] = max_after
    display(feature_ranges)

print('train set:')
check_feature_ranges(data_bf_norm=data_train, data_after_norm=data_train_norm)
print('test set:')
check_feature_ranges(data_bf_norm=data_test, data_after_norm=data_test_norm)

#save normalized data
data_train_norm.to_csv(f'{dataset_name}/data/train-norm.csv', index=False)
data_test_norm.to_csv(f'{dataset_name}/data/test-norm.csv', index=False)



##### split data

print('\n**********SPLIT DATA INTO X, y**********\n', file=f)

#split data into X, y (y=last column of dataset)
X_train = data_train_norm.iloc[:, 0:-1]
y_train = data_train_norm.iloc[:, -1]
X_test = data_test_norm.iloc[:, 0:-1]
y_test = data_test_norm.iloc[:, -1]

#save split data
X_train.to_csv(f'{dataset_name}/data/X-train-norm.csv', index=False)
y_train.to_csv(f'{dataset_name}/data/y-train-norm.csv', index=False)
X_test.to_csv(f'{dataset_name}/data/X-test-norm.csv', index=False)
y_test.to_csv(f'{dataset_name}/data/y-test-norm.csv', index=False)

#quick look
X=X_train
y=y_train
print('----- TRAIN -----', file=f)
print('X, shape:', X.shape, file=f)
print('y, shape:', y.shape, file=f)
print('#class1: ', sum(y), f', prop = {sum(y)/len(y)}', file=f)
print('#class0:', sum(y==0), f', prop = {sum(y==0)/len(y)}', file=f)

X=X_test
y=y_test
print('----- TEST -----', file=f)
print('X, shape:', X.shape, file=f)
print('y, shape:', y.shape, file=f)
print('#class1: ', sum(y), f', prop = {sum(y)/len(y)}', file=f)
print('#class0:', sum(y==0), f', prop = {sum(y==0)/len(y)}', file=f)



##### fix class imbalance: upsample smaller class in train set

print('\n**********UPSAMPLE SMALLER CLASS IN TRAIN SET**********\n', file=f)

#subset the two classes
class0 = data_train_norm[data_train_norm.iloc[:, -1] == 0]
class1 = data_train_norm[data_train_norm.iloc[:, -1] == 1]

#if there is class imbalance:
if len(class0) != len(class1):

    if len(class0) < len(class1):
        smaller_class = class0
        larger_class = class1
    else:
        smaller_class = class1
        larger_class = class0
        
###upsample smaller_class 
#find difference between smaller and larger class
n_upsample = len(larger_class) - len(smaller_class)
quotient, remainder = divmod(n_upsample, len(smaller_class))

#create multiples of the smaller_class set
smaller_class_upsampled = smaller_class.append([smaller_class]*quotient, ignore_index=True)
#then add random sample of 'remainder' number of points (sample without replacement)
smaller_class_upsampled = smaller_class_upsampled.append([smaller_class.sample(n=remainder)], ignore_index=True)

###combine upsampled smaller class + larger class
data_train_norm_upsampled = smaller_class_upsampled.append([larger_class], ignore_index=True)
#check that, after upsampling, both classes have the same number of points
class0_after = data_train_norm_upsampled[data_train_norm_upsampled.iloc[:, -1] == 0]
class1_after = data_train_norm_upsampled[data_train_norm_upsampled.iloc[:, -1] == 1]
print('After upsampling, classes are balanced:', len(class0_after) == len(class1_after), file=f)

#split data into X, y (y=last column of dataset)
X_train_norm_upsampled = data_train_norm_upsampled.iloc[:, 0:-1]
y_train_norm_upsampled = data_train_norm_upsampled.iloc[:, -1]

#save split data
X_train_norm_upsampled.to_csv(f'{dataset_name}/data/X-train-norm-upsampled.csv', index=False)
y_train_norm_upsampled.to_csv(f'{dataset_name}/data/y-train-norm-upsampled.csv', index=False)

#quick look
X=X_train_norm_upsampled
y=y_train_norm_upsampled
print('----- TRAIN, after upsampling -----', file=f)
print('X, shape:', X.shape, file=f)
print('y, shape:', y.shape, file=f)
print('#class1: ', sum(y), f', prop = {sum(y)/len(y)}', file=f)
print('#class0:', sum(y==0), f', prop = {sum(y==0)/len(y)}', file=f)



##### close print file

#close file
print('\n********** COMPLETE **********\n', file=f)
f.close()





