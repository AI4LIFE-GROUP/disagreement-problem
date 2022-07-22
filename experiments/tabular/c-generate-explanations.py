from xgboost import XGBClassifier
from models_torch import FFNN, LogisticRegressionNN

import pickle
import torch
import numpy as np
import pandas as pd

#explanation methods
from captum.attr import Lime, KernelShap, IntegratedGradients, Saliency, NoiseTunnel, InputXGradient
from lime import lime_tabular
import shap

import time
from datetime import datetime

import argparse

#parse arguments
parser = argparse.ArgumentParser(description='generate explanations')
parser.add_argument('--dataset_name', type = str, choices=['compas', 'german'], help='dataset name')

args = parser.parse_args()
dataset_name = args.dataset_name

#cat_vars: categorical variables used for explain_lime_og
if dataset_name == 'compas':
    cat_vars = ['two-year-recid', 'c-charge-degree', 'sex', 'race']
if dataset_name == 'german':
    cat_vars = ['status', 'credit-history', 'purpose', 'savings', 'employment-duration',
                'installment-rate', 'personal-status-sex', 'other-debtors', 'property', 
                'other-installment-plans', 'housing', 'job', 'telephone', 'foreign-worker']



##### load

f = open(f'{dataset_name}/explanations/notes_explanations.txt', 'w')
print('********** LOAD MODELS AND DATA **********\n', file=f, flush=True)

###load models
#logistic regression
model_filename = f'{dataset_name}/models/model_nn_logistic.pkl'
model_logistic = torch.load(model_filename)
#random forest
model_filename = f'{dataset_name}/models/model_rf.pkl'
model_rf = pickle.load(open(model_filename, 'rb'))
#gradient boosted tree
model_filename = f'{dataset_name}/models/model_gb.pkl'
model_gb = pickle.load(open(model_filename, 'rb'))
#neural network
model_filename = f'{dataset_name}/models/model_nn.pkl'
model_nn = torch.load(model_filename)

###load data (test set)
X_test = pd.read_csv(f'{dataset_name}/data/X-test-norm.csv').to_numpy() #(n_points_test, n_features) 2D
y_test = pd.read_csv(f'{dataset_name}/data/y-test-norm.csv').to_numpy().reshape(-1) #(n_points_test, ) 1D

#load data (train set)
X_train = pd.read_csv(f'{dataset_name}/data/X-train-norm.csv').to_numpy() #(n_points_train, n_features) 2D
#---> explain_lime_og() uses X_train for sample statistics (I don't use this, but still need to pass in as an argument)
#---> explain_ks_og() uses X_train as background dataset



##### set parameters

#number instances to explain
n = X_test.shape[0] #10
instances = X_test[0:n, :]

#list of samples
list_nsamples = [25, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000]



##### methods without nsample

def explain_vg(model, instances):
    model.zero_grad()
    method = Saliency(model)
    instances = torch.Tensor(instances)
    instances.requires_grad = True
    attr = method.attribute(instances)
    
    return attr.detach().numpy()


def explain_gxi(model, instances):
    model.zero_grad()
    method = InputXGradient(model)
    instances = torch.Tensor(instances)
    instances.requires_grad = True
    attr = method.attribute(instances)

    return attr.detach().numpy()



##### methods with nsample

#predict functions 
#--> use for rf and gb (non-torch model classes) when using captum
#--> input for explain_ks_og

# predict_fn: predicts probability of class1
#     in: instances, 2D np.array, n(=#datapoints) x p(=#features)
#     out: predictions, 1D np.array, n

def predict_fn_logistic(instances):
    return model_logistic.predict_proba(instances)[:, 1]

def predict_fn_rf(instances):
    return model_rf.predict_proba(instances)[:, 1]

def predict_fn_gb(instances):
    return model_gb.predict_proba(instances)[:, 1]

def predict_fn_nn(instances):
    return model_nn.predict_proba(instances)[:, 1]


#lime --- using original authors' implementation
def explain_lime_og(predict_fn, instances, training_data, nsamples, cat_vars=cat_vars, mode='classification', seed=12345):
    #get cat_idx (argument needed for lime explainer)
    var_names = pd.read_csv(f'{dataset_name}/data/X-train-norm.csv').columns
    cat_idxs = [list(var_names).index(var) for var in cat_vars]
    
    #create lime explainer
    explainer = lime_tabular.LimeTabularExplainer(training_data=training_data, 
                                                  mode=mode, 
                                                  discretize_continuous=False,
                                                  sample_around_instance=True,
                                                  random_state=seed, 
                                                  categorical_features=cat_idxs)
    #explain each data point in 'instances'
    exps = []
    num_feat = instances.shape[1]
    for x in instances:
        exp = explainer.explain_instance(x, predict_fn=predict_fn, num_samples=nsamples, num_features=num_feat).local_exp[1]
        #format explanations
        exp = sorted(exp, key=lambda tup: tup[0])
        exp = [t[1] for t in exp]
        exps.append(exp)
    
    #save explanations
    exps = np.array(exps)
    
    return exps #n x p (same dimensions as 'instances')


#kernelshap function --- using original authors' implementation

#kernelshap function
def explain_ks_og(predict_fn, instances, background_data, nsamples):
    '''
    #in
    #predict_fn: function that takes in an instance and outputs predictions
    #instances: instances to explain, [n_points, n_features]
    #background_data: background dataset, [n_points_in_background_dataset, n_features]
    #nsamples: number of perturbations to use, integer
    #out
    #shap_values: feature attributions, [n_points, n_features] (same dimensions as 'instances')
    '''
    #run kernelshap
    explainer = shap.KernelExplainer(model=predict_fn, data=background_data)
    shap_values = explainer.shap_values(X=instances, nsamples=nsamples)
    return shap_values


#integrated gradients
def explain_ig(model, instances, nsamples):
    model.zero_grad()
    method = IntegratedGradients(model)
    instances = torch.Tensor(instances)
    attr = method.attribute(instances, n_steps=nsamples)
    return attr.numpy()


#smoothgrad
def explain_sg(model, instances, nsamples):
    model.zero_grad()
    method = NoiseTunnel(Saliency(model))
    instances = torch.Tensor(instances)
    attr = method.attribute(instances, nt_type='smoothgrad', nt_samples=nsamples)
    return attr.numpy()



##### generate explanations

print('\n********** GENERATE EXPLANATIONS **********\n', file=f, flush=True)



##### vg, gxi

#vg, gxi
#run on logistic and nn models, no sample size

torch.manual_seed(123)
dict_methods = {'vg': explain_vg, 
                'gxi': explain_gxi}
dict_models = {'logistic': model_logistic, 
               'nn': model_nn}

#for each method:
for method_name, method in dict_methods.items():
    #for each model:
    for model_name, model in dict_models.items():
        #get explanations
        expl = method(model, instances)
        #save explanations
        filename = f'{dataset_name}/explanations/expl_{method_name}_{model_name}.pkl'
        pickle.dump(expl, open(filename, 'wb'))



##### ig, sg, lime (captum + og), ks (captum + og)

#create function for iterating over list of sample sizes
def get_explanations_many_nsamples(dict_methods, dict_models, list_nsamples):
    #for each method:
    for method_name, method in dict_methods.items():
        print('-'*20, file=f, flush=True)
        print('-'*20)
        
        #for each model:
        for model_name, model in dict_models.items():
            print(f'{method_name}, {model_name}', file=f, flush=True)
            print(f'{method_name}, {model_name}')

            #for each sample size:
            for nsample in list_nsamples:
                print(f'   nsamples={nsample}', file=f, flush=True)
                print(f'   nsamples={nsample}')
                start = time.time()
                print(f'      start: {datetime.now()}', file=f, flush=True)
                print(f'      start: {datetime.now()}')

                #generate explanation for model-explanationmethod-samplesize combo
                if method_name=='lime_og' or method_name=='ks_og':
                    expl = method(model, instances, X_train, nsample)
                else:
                    expl = method(model, instances, nsample)

                #save explanation for this combo
                expl_filepath = f'{dataset_name}/explanations/per_nsample/expl_{method_name}_{model_name}_{nsample}.pkl'
                pickle.dump(expl, open(expl_filepath, 'wb'))
                
                stop = time.time()
                print(f'      stop: {datetime.now()}', file=f, flush=True)
                print(f'      stop: {datetime.now()}')
                print(f'      duration: {(stop-start)/60} min', file=f, flush=True)
                print(f'      duration: {(stop-start)/60} min')



#ig, sg
#run on logistic and nn models, varying sample size

torch.manual_seed(123)
dict_methods = {'ig': explain_ig, 
                'sg': explain_sg
               }
dict_models = {'logistic': model_logistic, 
               'nn': model_nn}

#get explanations
get_explanations_many_nsamples(dict_methods, dict_models, list_nsamples)



#lime, ks ---> original authors implementation
#run on all models, varying sample size

np.random.seed(123)

###lime_og
dict_methods_limeog = {'lime_og': explain_lime_og}
predict_fns_limeog = {'logistic': model_logistic.predict_proba, 
               'gb': model_gb.predict_proba, 
               'rf': model_rf.predict_proba, 
               'nn': model_nn.predict_proba}
#get explanations
get_explanations_many_nsamples(dict_methods=dict_methods_limeog, 
                               dict_models=predict_fns_limeog, 
                               list_nsamples=list_nsamples)

###ks_og
dict_methods_ksog = {'ks_og': explain_ks_og}
predict_fns_ksog = {'logistic': predict_fn_logistic, 
               'gb': predict_fn_gb, 
               'rf': predict_fn_rf, 
               'nn': predict_fn_nn}
#get explanations
get_explanations_many_nsamples(dict_methods=dict_methods_ksog, 
                               dict_models=predict_fns_ksog, 
                               list_nsamples=list_nsamples)



##### close file


#close file
print('\n********** COMPLETE **********\n', file=f, flush=True)
f.close()


