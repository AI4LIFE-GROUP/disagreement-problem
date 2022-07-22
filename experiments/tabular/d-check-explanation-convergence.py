import numpy as np
import pickle

from metrics import compare_attr, compare_feature_ranks, rankcorr, pairwise_rank_agreement, agreement_fraction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import argparse

#parse arguments
parser = argparse.ArgumentParser(description='check convergence')
parser.add_argument('--dataset_name', type = str, choices=['compas', 'german'], help='dataset name')

args = parser.parse_args()
dataset_name = args.dataset_name



##### load explanations

def load_expl_dict_per_method(method_name, list_model_names, 
                              list_nsamples=[25, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000]):
    '''
    method_name: string, ex. 'lime'
    list_model_names: list of strings, ex. ['logistic', 'rf', 'gb', 'nn']
    '''
    expl_dict_all_models = {}

    for model_name in list_model_names:
        #create expl_dict for single model
        expl_dict_one_model = {}
        for nsample in list_nsamples:
            expl_dict_one_model[nsample] = pickle.load(open(f'{dataset_name}/explanations/per_nsample/expl_{method_name}_{model_name}_{nsample}.pkl', 'rb')) 

        #store expl_dict for single model
        expl_dict_all_models[model_name] = expl_dict_one_model
    
    return expl_dict_all_models


#load lime+ks explanations, 4 models
list_model_names = ['logistic', 'rf', 'gb', 'nn']
expl_lime_all_models = load_expl_dict_per_method(method_name='lime', list_model_names=list_model_names)
expl_ks_all_models = load_expl_dict_per_method(method_name='ks', list_model_names=list_model_names)

expl_lime_og_all_models = load_expl_dict_per_method(method_name='lime_og', list_model_names=list_model_names)
expl_ks_og_all_models = load_expl_dict_per_method(method_name='ks_og', list_model_names=list_model_names)

#load ig+sg explanations, 2 models
list_model_names = ['logistic', 'nn']
expl_ig_all_models = load_expl_dict_per_method(method_name='ig', list_model_names=list_model_names)
expl_sg_all_models = load_expl_dict_per_method(method_name='sg', list_model_names=list_model_names)



##### calculate convergence metrics

#store explanations in dictionaries
dict_expl_all_models = {'lime': expl_lime_all_models,
                       'ks': expl_ks_all_models,
                       'ig': expl_ig_all_models,
                       'sg': expl_sg_all_models,
                       'lime_og': expl_lime_og_all_models,
                       'ks_og': expl_ks_og_all_models}

dict_title_methods = {'lime': 'LIME (Captum)',
                     'ks': 'KernelSHAP (Captum)',
                     'ig': 'Integrated Gradients',
                     'sg': 'SmoothGrad',
                     'lime_og': 'LIME (original library)' , 
                     'ks_og': 'KernelSHAP (original library)'}

dict_title_models = {'logistic': 'Logistic Regression',
                     'rf': 'Random Forest',
                     'gb': 'Gradient-Boosted Tree',
                     'nn': 'Neural Network'}


def format_dict_for_boxplot(distr_dict):
    df_list = []
    ns = list(distr_dict.keys())
    for i in range(0, len(ns)):
        df = pd.DataFrame(distr_dict[ns[i]], columns=['gap_metric'])
        df['n'] = ns[i]
        df_list.append(df)
    df_full = pd.concat(df_list)
    return df_full


#calculate metrics and check their convergence for each method-model-metric combo

#for each method...
for method_name, expl_all_models in dict_expl_all_models.items():
    
    #for each model...
    for model_name, expl_dict in expl_all_models.items():
        print(f'calculating metrics for {method_name}, {model_name}:')

        #create empty dictionaries for storage
        gaps_l2_attr = {} #L2 distance of feature attributions
        gaps_l2_ranks = {} #L2 distance of feature ranks
        gaps_fa = {} #feature agreement
        gaps_ra = {} #rank agreement
        gaps_sa = {} #sign agreement
        gaps_sra = {} #signed rank agreement
        gaps_rc = {} #rank correlation
        gaps_pra = {} #paired rank agreement
        
        #for each nsample...
        list_nsamples = list(expl_dict.keys())
        for i in range(1, len(list_nsamples)): #start at 1, so that 1st is compared with 0th
            print(f'   n={list_nsamples[i]}')
            
            #get two sets of attributions
            nsample_curr = list_nsamples[i]
            nsample_prev = list_nsamples[i-1]
            attr_curr = expl_dict[nsample_curr]
            attr_prev = expl_dict[nsample_prev]
            n_features = attr_curr.shape[1]
            
            #calculate metrics for attributions
            gaps_l2_attr[nsample_curr] = compare_attr(attr_curr, attr_prev)
            gaps_l2_ranks[nsample_curr] = compare_feature_ranks(attr_curr, attr_prev)
            gaps_fa[nsample_curr] = agreement_fraction(attrA=attr_curr, attrB=attr_prev, k=n_features, metric='feature')
            gaps_ra[nsample_curr] = agreement_fraction(attrA=attr_curr, attrB=attr_prev, k=n_features, metric='rank')
            gaps_sa[nsample_curr] = agreement_fraction(attrA=attr_curr, attrB=attr_prev, k=n_features, metric='sign')
            gaps_sra[nsample_curr] = agreement_fraction(attrA=attr_curr, attrB=attr_prev, k=n_features, metric='signedrank')
            gaps_rc[nsample_curr] = rankcorr(attr_curr, attr_prev)
            gaps_pra[nsample_curr] = pairwise_rank_agreement(attr_curr, attr_prev)   

        #format dictionaries into df for plotting
        df_l2_attr = format_dict_for_boxplot(gaps_l2_attr)
        df_l2_ranks = format_dict_for_boxplot(gaps_l2_ranks)
        df_fa = format_dict_for_boxplot(gaps_fa)
        df_ra = format_dict_for_boxplot(gaps_ra)
        df_sa = format_dict_for_boxplot(gaps_sa)
        df_sra = format_dict_for_boxplot(gaps_sra)
        df_rc = format_dict_for_boxplot(gaps_rc)
        df_pra = format_dict_for_boxplot(gaps_pra)
        
        ###plot boxplots
        fig, axes = plt.subplots(2, 4, figsize =(20, 8))
        #L2 of attributions
        plot_l2_attr = sns.boxplot(x='n', y='gap_metric', data=df_l2_attr, color='cornflowerblue', ax=axes[0, 0])
        plot_l2_attr.set(xlabel='Number of Samples (n)', ylabel='L2(FA at Current n, FA at Previous n)', title='Feature Attribution (FA)')
        #L2 of feature importances
        plot_l2_ranks = sns.boxplot(x='n', y='gap_metric', data=df_l2_ranks, color='cornflowerblue', ax=axes[0, 1])
        plot_l2_ranks.set(xlabel='Number of Samples (n)', ylabel='L2(FR at Current n, FR at Previous n)', title='Feature Rank (FR)')
        #feature agreement
        plot_fa = sns.boxplot(x='n', y='gap_metric', data=df_fa, color='cornflowerblue', ax=axes[0, 2])
        plot_fa.set(xlabel='Number of Samples (n)', ylabel='Feature Agreement\n(Current vs. Previous n)', title='Feature Agreement')
        #rank agreement
        plot_ra = sns.boxplot(x='n', y='gap_metric', data=df_ra, color='cornflowerblue', ax=axes[0, 3])
        plot_ra.set(xlabel='Number of Samples (n)', ylabel='Rank Agreement\n(Current vs. Previous n)', title='Rank Agreement')
        #sign agreement
        plot_sa = sns.boxplot(x='n', y='gap_metric', data=df_sa, color='cornflowerblue', ax=axes[1, 0])
        plot_sa.set(xlabel='Number of Samples (n)', ylabel='Sign Agreement\n(Current vs. Previous n)', title='Sign Agreement')
        #signed rank agreement
        plot_sra = sns.boxplot(x='n', y='gap_metric', data=df_sra, color='cornflowerblue', ax=axes[1, 1])
        plot_sra.set(xlabel='Number of Samples (n)', ylabel='Signed Rank Agreement\n(Current vs. Previous n)', title='Signed Rank Agreement')
        #rank correlation
        plot_rc = sns.boxplot(x='n', y='gap_metric', data=df_rc, color='cornflowerblue', ax=axes[1, 2])
        plot_rc.set(xlabel='Number of Samples (n)', ylabel='Rank Correlation\n(Current vs. Previous n)', title='Rank Correlation')
        #pairwise rank agreement
        plot_pra = sns.boxplot(x='n', y='gap_metric', data=df_pra, color='cornflowerblue', ax=axes[1, 3])
        plot_pra.set(xlabel='Number of Samples (n)', ylabel='Pairwise Rank Agreement\n(Current vs. Previous n)', title='Pairwise Rank Agreement')
        #save plot
        plot_title=f'Convergence Check: {dict_title_methods[method_name]}, {dict_title_models[model_name]}'
        plot_path=f'{dataset_name}/figures/convergence/{method_name}_{model_name}.png'
        fig.suptitle(plot_title)
        fig.tight_layout()
        fig.savefig(plot_path, facecolor='white', transparent=False)

        
