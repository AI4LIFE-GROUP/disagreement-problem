import numpy as np
import pandas as pd

#models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from models_torch import LogisticRegressionNN, FFNN, FFNNA, FFNNB, FFNNC

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

#model evaluation
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, roc_curve, auc

import pickle

import argparse

#parse arguments
parser = argparse.ArgumentParser(description='train models')
parser.add_argument('--dataset_name', type = str, choices=['compas', 'german'], help='dataset name')
parser.add_argument('--mode_train', type = bool, choices=[True, False], help='whether to train (train and save model) or evaluate (load saved model and evaluate performance) model')

args = parser.parse_args()
dataset_name = args.dataset_name
mode_train = args.mode_train



##### load data

if mode_train:
    f = open(f'{dataset_name}/models/notes_models_train.txt', 'w')
if not mode_train:
    f = open(f'{dataset_name}/models/notes_models_evaluate.txt', 'w')

print('********** LOAD DATA **********\n', file=f)

X_train = pd.read_csv(f'{dataset_name}/data/X-train-norm-upsampled.csv').to_numpy() #(n_points_train, n_features) 2D
y_train = pd.read_csv(f'{dataset_name}/data/y-train-norm-upsampled.csv').to_numpy().reshape(-1) #(n_points_train, ) 1D

X_test = pd.read_csv(f'{dataset_name}/data/X-test-norm.csv').to_numpy() #(n_points_test, n_features) 2D
y_test = pd.read_csv(f'{dataset_name}/data/y-test-norm.csv').to_numpy().reshape(-1) #(n_points_test, ) 1D



##### function to evaluate model performance

#evaluate model for one dataset
def evaluate_model(y_true, y_pred, y_prob_class1):
    #overall
    print('acccuracy:', accuracy_score(y_true, y_pred), file=f)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_class1) #y_prob
    print('AUC:', auc(fpr, tpr), file=f)
    
    #class 1
    print('\n***class 1***', file=f)
    print('acccuracy:', accuracy_score(y_true=y_true[y_true==1], y_pred=y_pred[y_true==1]), file=f)
    pos_label = 1
    print('recall:', recall_score(y_true, y_pred, pos_label=pos_label), file=f)
    print('precision:', precision_score(y_true, y_pred, pos_label=pos_label), file=f)
    print('F1 score:', f1_score(y_true, y_pred, pos_label=pos_label), file=f)
    print('AP:', average_precision_score(y_true, y_pred, pos_label=pos_label), file=f)
    
    #class 0
    print('\n***class 0***', file=f)
    print('acccuracy:', accuracy_score(y_true=y_true[y_true==0], y_pred=y_pred[y_true==0]), file=f)
    pos_label = 0
    print('recall:', recall_score(y_true, y_pred, pos_label=pos_label), file=f)
    print('precision:', precision_score(y_true, y_pred, pos_label=pos_label), file=f)
    print('F1 score:', f1_score(y_true, y_pred, pos_label=pos_label), file=f)
    print('AP:', average_precision_score(y_true, y_pred, pos_label=pos_label), file=f)


#evaluate model for train and test sets
def evaluate_model_train_and_test_sets(model_filename, X_train, y_train, X_test, y_test, load=['pickle', 'torch']):
    #load saved model
    if load=='pickle':
        model = pickle.load(open(model_filename, 'rb'))
    if load=='torch':
        model = torch.load(model_filename)
    
    #evaluate model
    #training set
    X=X_train
    y=y_train
    print('----- TRAIN -----', file=f)
    evaluate_model(y_true=y, 
                   y_pred=model.predict(X), 
                   y_prob_class1=model.predict_proba(X)[:, 1])
    #test set
    X=X_test
    y=y_test
    print('\n----- TEST -----', file=f)
    evaluate_model(y_true=y, 
                   y_pred=model.predict(X), 
                   y_prob_class1=model.predict_proba(X)[:, 1])



##### model 1: logistic regression

#model 1: logistic regression
print('********** MODEL 1: LOGISTIC REGRESSION **********\n', file=f)
model_filename = f'{dataset_name}/models/model_logistic.pkl'

#train model
if mode_train:
    #fit model
    model_logistic = LogisticRegression(penalty='none').fit(X_train, y_train)
    #save model
    pickle.dump(model_logistic, open(model_filename, 'wb'))
    
#evaluate model
if not mode_train:
    evaluate_model_train_and_test_sets(model_filename, X_train, y_train, X_test, y_test, load='pickle')



##### model 2: gradient boosted tree

#model 2: gradient boosted tree
print('\n********** MODEL 2: GRADIENT BOOSTED TREE **********\n', file=f)
model_filename = f'{dataset_name}/models/model_gb.pkl'

#train model
if mode_train:
    #fit model
    model_gb = XGBClassifier(n_estimators=50, random_state=12345, use_label_encoder=False, eval_metric='logloss')
    #use_label_encoder=False, remove warning (no impact on model performance)
    #eval_metric='logloss', remove warning (this is the default, but need to specify)
    model_gb.fit(X_train, y_train)
    #save model
    pickle.dump(model_gb, open(model_filename, 'wb'))
    
#evaluate model
if not mode_train:
    evaluate_model_train_and_test_sets(model_filename, X_train, y_train, X_test, y_test, load='pickle')



##### model 3: random forest

#model 3: random forest
print('\n********** MODEL 3: RANDOM FOREST **********\n', file=f)
model_filename = f'{dataset_name}/models/model_rf.pkl'

#train model
if mode_train:
    #fit model
    model_rf = RandomForestClassifier(n_estimators=50, random_state=12345)
    model_rf.fit(X_train, y_train)
    #save model
    pickle.dump(model_rf, open(model_filename, 'wb'))
    
#evaluate model
if not mode_train:
    evaluate_model_train_and_test_sets(model_filename, X_train, y_train, X_test, y_test, load='pickle')



##### model 4: neural network

#model 4: neural network
print('\n********** MODEL 4: NEURAL NETWORK **********\n', file=f)
model_filename = f'{dataset_name}/models/model_nn.pkl'

#create MyDataset class
class MyDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
    
#create datasets
train_ds = MyDataset(X_train, y_train)
test_ds = MyDataset(X_test, y_test)

#create dataloaders
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)


#train model

#function to calculate accuracy
def compute_n_correct(y_true, pred_prob): 
    y_pred = (pred_prob.squeeze()>0.5)*1
    return sum(y_pred == y_true)

#function to train ffnn model
def train_ffnn(model, model_filename):
    #set training parameters
    n_epochs = 20
    seed = 12345
    torch.manual_seed(seed)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    
    train_losses = []
    train_accs = []
    
    #loop through epochs and batches
    for epoch in range(n_epochs):
        running_loss = 0
        running_n_correct = 0

        for batch_input, batch_output in train_dl:
            batch_input = batch_input.type(torch.FloatTensor)
            batch_output = batch_output.type(torch.FloatTensor)
            #forward pass: compute model output and loss
            preds = model(batch_input)
            loss = loss_fn(preds.squeeze(), batch_output)
            #backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #store metrics
            running_loss += loss.cpu().item()
            n_correct = compute_n_correct(y_true=batch_output, pred_prob=preds)
            running_n_correct += n_correct

        train_losses.append(running_loss / len(train_ds))
        train_accs.append(running_n_correct / len(train_ds))
        print('-'*20, file=f)
        print(f'Epoch {epoch+1}/{n_epochs} Train Loss: {running_loss / len(train_ds)}', file=f)
        print(f'Epoch {epoch+1}/{n_epochs} Train Accuracy: {running_n_correct / len(train_ds)}', file=f)

    #save model
    torch.save(model, model_filename)


#train model
if mode_train:
    train_ffnn(model= FFNN(input_size=X_train.shape[1], hidden_size=50), 
               model_filename=f'{dataset_name}/models/model_nn.pkl')

#evaluate model
if not mode_train:
    evaluate_model_train_and_test_sets(model_filename, X_train, y_train, X_test, y_test, load='torch') #evaluate model



##### model 5: logistic regression as neural network
# to calculate gradients for gradient-based explanation methods

#model 5: logistic regression as neural network
print('\n********** MODEL 5: LOGISTIC REGRESSION AS NEURAL NETWORK **********\n', file=f)
model_filename = f'{dataset_name}/models/model_nn_logistic.pkl'

#logistic regression parameters, from logistic regression
model_logistic = pickle.load(open(f'{dataset_name}/models/model_logistic.pkl', 'rb'))
print(model_logistic.coef_)
print(model_logistic.intercept_)


#change NN model weights to logistic regression coefficient

if mode_train:
    #instantiate model
    model_nn_logistic = LogisticRegressionNN(input_size=X_train.shape[1])

    #change NN model weights
    lr_coefs = torch.tensor(model_logistic.coef_, requires_grad=True).type(torch.FloatTensor)
    lr_intercept = torch.tensor(model_logistic.intercept_, requires_grad=True).type(torch.FloatTensor)

    model_nn_logistic.linear_layer.weight = nn.Parameter(lr_coefs)
    model_nn_logistic.linear_layer.bias = nn.Parameter(lr_intercept)

    print(model_nn_logistic.linear_layer.weight)
    print(model_nn_logistic.linear_layer.bias)

    #save model
    torch.save(model_nn_logistic, model_filename)

#evaluate model
if not mode_train:
    evaluate_model_train_and_test_sets(model_filename, X_train, y_train, X_test, y_test, load='torch')



##### FFNN A, B, C

#train model
if mode_train:
    print('\n********** FFNN A **********\n', file=f)
    train_ffnn(model=FFNNA(input_size=X_train.shape[1], hidden_size=10), 
               model_filename=f'{dataset_name}/models/model_nnA.pkl')
    
    print('\n********** FFNN B **********\n', file=f)
    train_ffnn(model=FFNNB(input_size=X_train.shape[1], hidden_size=10), 
               model_filename=f'{dataset_name}/models/model_nnB.pkl')
    
    print('\n********** FFNN C **********\n', file=f)
    train_ffnn(model=FFNNC(input_size=X_train.shape[1], hidden_size=10), 
               model_filename=f'{dataset_name}/models/model_nnC.pkl')

#evaluate model
if not mode_train:
    print('\n********** FFNN A **********\n', file=f)
    evaluate_model_train_and_test_sets(f'{dataset_name}/models/model_nnA.pkl', X_train, y_train, X_test, y_test, load='torch')
    print('\n********** FFNN B **********\n', file=f)
    evaluate_model_train_and_test_sets(f'{dataset_name}/models/model_nnB.pkl', X_train, y_train, X_test, y_test, load='torch')
    print('\n********** FFNN C **********\n', file=f)
    evaluate_model_train_and_test_sets(f'{dataset_name}/models/model_nnC.pkl', X_train, y_train, X_test, y_test, load='torch')
    
    

##### close file

#close file
print('\n********** COMPLETE **********\n', file=f)
f.close()


