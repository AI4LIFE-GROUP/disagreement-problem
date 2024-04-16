import numpy as np
import pickle
import itertools
from metrics import rankcorr, pairwise_rank_agreement, agreement_fraction, modified_rank_agreement

from scipy.stats import sem

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


import argparse

#parse arguments
parser = argparse.ArgumentParser(description='compare explanations')
parser.add_argument('--dataset_name', type = str, choices=['compas', 'german'], help='dataset name')

args = parser.parse_args()
dataset_name = args.dataset_name

#list_ks: list of k-values to use for top-k metrics
if dataset_name == 'compas':
    list_ks = [1, 4, 5, 7]
if dataset_name == 'german':
    list_ks = [7, 14, 20]



##### load explanations

#function to load explanations
def load_expl(method_name, list_model_names, nsample=2000):
    '''
    method_name: string, name of explanation method
    nsample: integer, 
    list_model_names: list of model names (strings) for which to load explanations, ex. ['logistic', 'rf', 'gb', 'nn']
    '''
    #methods with nsample parameter
    if method_name in ['lime', 'lime_og', 'ks', 'ks_og', 'ig', 'sg']:
        filenames = {model_name: f'{dataset_name}/explanations/per_nsample/expl_{method_name}_{model_name}_{nsample}.pkl' for model_name in list_model_names}
    #methods without nsample parameter
    if method_name in ['vg', 'gxi']:
        filenames = {model_name: f'{dataset_name}/explanations/expl_{method_name}_{model_name}.pkl' for model_name in list_model_names}
    
    #load explanations
    expl_dict = {model_name: pickle.load(open(filenames[model_name], 'rb')) for model_name in list_model_names}

    return expl_dict


#pick explanations at convergence

###perturbation-based methods
list_model_names = ['logistic', 'rf', 'gb', 'nn']
expl_lime = load_expl(method_name='lime_og', list_model_names=list_model_names)
expl_ks = load_expl(method_name='ks_og', list_model_names=list_model_names)

###gradient-based methods
list_model_names = ['logistic', 'nn']
#with nsample parameter
expl_ig = load_expl(method_name='ig', list_model_names=list_model_names)
expl_sg = load_expl(method_name='sg', list_model_names=list_model_names)
#without nsample parameter
expl_vg = load_expl(method_name='vg', list_model_names=list_model_names)
expl_gxi = load_expl(method_name='gxi', list_model_names=list_model_names)


##### plotting functions

#dictionaries for plot titles/labels
dict_title_methods_heatmap = {'lime': 'LIME',
                              'ks': 'Kernel\nSHAP',
                              'vg': 'Grad',
                              'gxi': 'Grad*\nInput',
                              'ig': 'IntGrad',
                              'sg': 'Smooth\nGrad'}

dict_title_methods_boxplot = {'lime': 'LIME',
                              'ks': 'KernelSHAP',
                              'vg': 'Grad',
                              'gxi': 'Grad*Input',
                              'ig': 'IntGrad',
                              'sg': 'SmoothGrad'}

dict_title_metrics = {'feature': 'Feature agreement', 
                      'rank': 'Rank agreement', 
                      'sign': 'Sign agreement', 
                      'signedrank': 'Signed rank agreement',
                      'rc': 'Rank correlation', 
                      'pra': 'Pairwise rank agreement',
                       'ra' : 'Weighted rank agreement'}



def boxplot_metric_distr(method_pairs_distr, plot_path, metric, k):
    '''
    method_pairs_distr: dictionary of metric distribution for each method-pair (key=method-pair string, value=1D numpy array of metric distribution)
    plot_path: string, path to save plot
    k: integer, k-value for metrics that are top-k-based (used in title)
    metric: string, from list ['feature', 'rank', 'sign', 'signedrank', 'rc', 'pra']
    '''
    #boxplot
    fig, axes = plt.subplots(1, 1, figsize=(5.5, 5.5))
    axes.set(ylim=(-1.1 if metric=='rc' else -0.1, 1.1)) 
    
    bp = sns.boxplot(data=list(method_pairs_distr.values()), color='cornflowerblue', ax=axes)
    bp.set_xticklabels(list(method_pairs_distr.keys()), rotation=90)


    if metric in ['rc', 'pra']:
        plot_title=dict_title_metrics[metric]
    else:
        plot_title=f'{dict_title_metrics[metric]} (k = {k})'

    bp.set(xlabel='Pair of explanation methods', ylabel=dict_title_metrics[metric], title=plot_title)

    fig.tight_layout()
    fig.savefig(plot_path, facecolor='white', transparent=False, bbox_inches='tight', dpi=1200)



def heatmap_metric_avg(method_pairs_avg, method_pairs_sem, plot_path, k, metric, list_method_names):
    '''
    method_pairs_avg: numpy array [n_methods, n_methods] of mean of metric distribution for each method-pair
    method_pairs_sem: numpy array [n_methods, n_methods] of sem of metric distribution for each method-pair
    plot_path: string, path to save plot
    k: integer, k-value for metrics that are top-k-based (used in title)
    metric: string, from list ['feature', 'rank', 'sign', 'signedrank', 'rc', 'pra']
    list_method_names: list of abbreviated method names, ex. ['lime', 'ks'] to be converted into official method names    
    '''
    
    #mask = np.invert(np.tril(np.ones_like(corr_matrix, dtype=bool))) #mask for upper triangle
    cmap = sns.color_palette('vlag', as_cmap=True) #diverging colormap
    
    #x-axis and y-axis labels
    labels=[dict_title_methods_heatmap[method_name] for method_name in list_method_names]
    
    #heatmap
    plt.figure(figsize=(15, 7))
    sns.set(font_scale=1.5)
    sns.heatmap(method_pairs_avg, cmap=cmap, #mask=mask, 
                vmin=-1 if metric=='rc' else 0, vmax=1, center=0,
                xticklabels=labels, yticklabels=labels, annot=True, fmt='.3f',
                square=True, linewidths=.5, cbar_kws={'shrink': 0.995}) #annot_kws={'fontsize':'large'}
    plt.yticks(rotation=0)

    if metric in ['rc', 'pra']:
        plt.title(dict_title_metrics[metric])
    else:
        plt.title(f'{dict_title_metrics[metric]} (k = {k})')
    
    sem_min = method_pairs_sem.min().round(3)
    sem_max = method_pairs_sem.max().round(3)
    plt.figtext(x=0.385, y=0.00005, s=f'Standard errors: min={sem_min}, max={sem_max}', fontsize='medium')

    plt.savefig(plot_path, facecolor='white', transparent=False, bbox_inches='tight', dpi=2000)



##### calculate metrics to measure explanation disagreement

def calculate_and_plot_metrics(dict_model_expl_combos, metric, k):
    '''
    dict_model_expl_combos: dictionary with feature attributions for each model and each method (key=model_name (string), value={method_name: feature attribution [n_points, n_features]})
    metric: string, from list ['feature', 'rank', 'sign', 'signedrank', 'rc', 'pra']
    k: integer, used for top-k-based metrics
    '''
    
    for model_name, dict_expls in dict_model_expl_combos.items():
        print(f'Calculating metric for model={model_name}')

        ###calculate metric distribution for all method-pairs --> get distributions for boxplots
        n_methods = len(dict_expls)
        n_datapoints = list(dict_expls.values())[0].shape[0]
        metric_distr = np.zeros([n_methods, n_methods, n_datapoints]) #storage: 3D array to store full metric data --> each strip ([A, B, :]) is the distribution of the metric for method-pair A-B
        metric_distr_dict = {} #storage: dict to store distribution of metric for each method pair

        #for each method-pair...
        for idxA, idxB in itertools.combinations_with_replacement(range(n_methods), 2):
            #get feature attributions of methodA and methodB
            method_nameA = list(dict_expls.keys())[idxA]
            method_nameB = list(dict_expls.keys())[idxB]
            explA = dict_expls[method_nameA]
            explB = dict_expls[method_nameB]

            #calculate metric distribution
            if metric in ['feature', 'rank', 'sign', 'signedrank']: #4 agreement fraction metrics
                metric_distr_onemethodpair = agreement_fraction(explA, explB, k, metric)
            if metric=='rc':
                metric_distr_onemethodpair = rankcorr(explA, explB)
            if metric=='pra':
                metric_distr_onemethodpair = pairwise_rank_agreement(explA, explB)
            if metric == "ra":
                metric_distr_onemethodpair = modified_rank_agreement(explA, explB, k)

            #store metric distribution
            metric_distr[idxA, idxB, :] = metric_distr_onemethodpair
            metric_distr[idxB, idxA, :] = metric_distr_onemethodpair

            #store metric distribution in dictionary
            if idxA != idxB:
                metric_distr_dict[f'{dict_title_methods_boxplot[method_nameA]} vs. {dict_title_methods_boxplot[method_nameB]}'] = metric_distr_onemethodpair

        ###plot boxplot
        plot_folder = f'{dataset_name}/figures/disagreement/ra_plots/'
        plot_name_boxplot = f'{model_name}_{metric}_distr.png' if metric in ['rc', 'pra'] else f'{model_name}_{metric}_k{k}_distr.png'
        plot_path= plot_folder + plot_name_boxplot
        matplotlib.rc_file_defaults() #make background white for plots
        boxplot_metric_distr(metric_distr_dict, plot_path, metric, k)

        ###calculate metric average and sem for all method-pairs --> form matrix for heatmap
        metric_avg = metric_distr.mean(axis=2) #[n_methods, n_methods]
        metric_sem = sem(metric_distr, axis=2) #[n_methods, n_methods]

        ###plot heatmap
        plot_name_heatmap = f'{model_name}_{metric}_avg.png' if metric in ['rc', 'pra'] else f'{model_name}_{metric}_k{k}_avg.png'
        plot_path = plot_folder + plot_name_heatmap
        heatmap_metric_avg(metric_avg, metric_sem, plot_path, k, metric, list_method_names=list(dict_expls.keys()))



##### calculate metrics: run calculate_and_plot_metrics()

#create dictionary of feature attributions organized by models
dict_model_expl_combos = {}

#models that have gradients
for model_name in ['logistic', 'nn']:
    dict_model_expl_combos[model_name] = {'lime': expl_lime[model_name], 
                                          'ks': expl_ks[model_name], 
                                          'vg': expl_vg[model_name], 
                                          'gxi': expl_gxi[model_name], 
                                          'ig': expl_ig[model_name], 
                                          'sg': expl_sg[model_name]}
#models that do not have gradients
for model_name in ['rf', 'gb']:
    dict_model_expl_combos[model_name] = {'lime': expl_lime[model_name], 
                                          'ks': expl_ks[model_name]}


# #non top-k-based metrics
# for metric in ['rc', 'pra']:
#     print(f'***** metric={metric}*****')
#     calculate_and_plot_metrics(dict_model_expl_combos, metric, k=None) 



#top-k-based metrics
for metric in ['ra', 'rank']: #['feature', 'rank', 'sign', 'signedrank', 'ra']:
    for k in list_ks:
        print(f'***** metric={metric}, k={k}*****')
        calculate_and_plot_metrics(dict_model_expl_combos, metric, k)


