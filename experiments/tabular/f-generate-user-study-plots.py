import pickle
import numpy as np

import itertools
from metrics import rankcorr, pairwise_rank_agreement, agreement_fraction

from scipy.stats import spearmanr, percentileofscore

import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset_name='compas'


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


#pick explanations at convergence for nn model

#perturbation-based methods
list_model_names = ['nn']
expl_lime = load_expl(method_name='lime_og', list_model_names=list_model_names)
expl_ks = load_expl(method_name='ks_og', list_model_names=list_model_names)

#gradient-based methods
expl_vg = load_expl(method_name='vg', list_model_names=list_model_names) #without nsample parameter
expl_gxi = load_expl(method_name='gxi', list_model_names=list_model_names)
expl_ig = load_expl(method_name='ig', list_model_names=list_model_names) #with nsample parameter
expl_sg = load_expl(method_name='sg', list_model_names=list_model_names)


expl_methods_names = {0: 'LIME',
                      1: 'KernelSHAP', 
                      2: 'Gradient',
                      3: 'Gradient*Input',
                      4: 'Integrated Gradients',
                      5: 'SmoothGrad'}
m='nn' #model
expl_dict = {0: expl_lime[m],
             1: expl_ks[m], 
             2: expl_vg[m],
             3: expl_gxi[m],
             4: expl_ig[m],
             5: expl_sg[m]}


##### calculate disagreement for each point, metric, and method-pair

n_points = expl_dict[0].shape[0] #number of points in test set (number of data points for which we have explanations)
n_methods = 6
factorial = np.math.factorial
n_methodpairs = int(factorial(n_methods)/( factorial(n_methods-2) * factorial(2) )) #n_methods choose 2
n_metrics = 6
n_features = expl_dict[0].shape[1] #number of features in compas dataset = 7

#set up matrix to fill in
metrics_pointxmetricxpair = np.zeros([n_points, n_metrics, n_methodpairs])

#for each method pair
for idx_method_pair, (idx_methodA, idx_methodB) in enumerate(itertools.combinations(range(n_methods), 2)):
    print(f'method pair {idx_method_pair}: {expl_methods_names[idx_methodA]} ({idx_methodA}), {expl_methods_names[idx_methodB]}({idx_methodB})')
    
    #get attributions of this method pair
    attrA = expl_dict[idx_methodA]
    attrB = expl_dict[idx_methodB]
    
    #calculate disagreement based on each of the 6 metrics
    rc = rankcorr(attrA, attrB) #rank correlation, all features
    pra = pairwise_rank_agreement(attrA, attrB) #pairwise rank agreement, all features
    fa = agreement_fraction(attrA, attrB, k=n_features, metric='feature') #feature agreement, k=all features
    ra = agreement_fraction(attrA, attrB, k=n_features, metric='rank') #rank agreement, k=all features
    sa = agreement_fraction(attrA, attrB, k=n_features, metric='sign') #sign agreement, k=all features
    sra = agreement_fraction(attrA, attrB, k=n_features, metric='signedrank') #signed rank agreement, k=all features

    #fill in 2D slice of metrics_per_point (3D matrix)
    metrics_pointxmetricxpair[:, :, idx_method_pair] = np.stack([rc, pra, fa, ra, sa, sra], axis=1) #n_points x n_metrics



##### for each method pair, select point with high disagreement based on all 6 metrics

#calculate percentiles for each set of 'n_points', i.e. for each combination along the other two dimensions
ptiles = np.zeros([n_points, n_metrics, n_methodpairs])

for idx_metric in range(n_metrics):
    for idx_methodpair in range(n_methodpairs):
        #distr=distribution of metric values for a given metric for a given method-pair
        metric_distr = metrics_pointxmetricxpair[:, idx_metric, idx_methodpair]
        
        #for each value in distr, calculate percentile of that value
        ptiles[:, idx_metric, idx_methodpair] = [(metric_distr < point_value).mean() for point_value in metric_distr]



#####for each methodpair, find a point meeting the criteria

random.seed(12345)

dict_userstudy_points_info = {}

#for each methodpair...
for idx_methodpair in range(n_methodpairs):
    #get 2D array of percentiles for this methodpair
    ptiles_pointsxmetrics = ptiles[:, :, idx_methodpair]
    
    #for possible percentile cutoffs (starting from lowest possible cutoff)
    for ptile_cutoff_lower in np.arange(0, 1, 0.01):
        ptile_cutoff_upper = ptile_cutoff_lower + 0.01

        points_meeting_criteria_lower = (ptiles_pointsxmetrics < ptile_cutoff_lower).sum(axis=1) == n_metrics #1D array of booleans: points with all 6 metrics under ptile_cutoff
        points_meeting_criteria_upper = (ptiles_pointsxmetrics < ptile_cutoff_upper).sum(axis=1) == n_metrics #1D array of booleans: points with all 6 metrics under ptile_cutoff
        
        #find lowest ptile cutoff such that there are a non-zero number of points meeting the criteria --> aka find 'ptile_cutoff_upper' (yes UPPER)
        if np.sum(points_meeting_criteria_lower)==0 and np.sum(points_meeting_criteria_upper)>0:
            #get idxs of points meeting criteria
            idx_points_all, = np.where(points_meeting_criteria_upper) #idx of all points meeting criteria
            
            #select a point to use for user study
            if len(idx_points_all) > 1: #if more than one point meets the criteria, select a point with lowest average percentile
                idx_of_idxs_with_lowest_avg_ptile = ptiles[idx_points_all, :, idx_methodpair].mean(axis=1).argmin()
                idx_point = idx_points_all[idx_of_idxs_with_lowest_avg_ptile]
                
            else: #if only one point meets the criteria, select that point
                idx_point = idx_points_all[0]
            
            #store results
            dict_userstudy_points_info[idx_methodpair] = {'ptile_cutoff': ptile_cutoff_upper,
                                                         'num_points_meeting_criteria': len(idx_points_all),
                                                         'idx_points_meeting_criteria': idx_points_all,
                                                         'idx_point_selected': idx_point}
            #print results
            print(f'idx_methodpair: {idx_methodpair}')
            print(f'    ptile_cutoff: {ptile_cutoff_upper}')
            print(f'    num_points_meeting_criteria: {len(idx_points_all)}')
            print(f'    idx_point selected: {idx_point}')
            for point in idx_points_all:
                print(f'    metric values: point {point}, {np.around(ptiles[point, :, idx_methodpair], 4)}')

            break



##### generate user study plots

#for each method pair
for idx_method_pair, (idx_methodA, idx_methodB) in enumerate(itertools.combinations(range(n_methods), 2)):
    
    #get attributions of this method pair for this data point
    idx_point = dict_userstudy_points_info[idx_method_pair]['idx_point_selected']
    expl1 = expl_dict[idx_methodA][idx_point, :]
    expl2 = expl_dict[idx_methodB][idx_point, :]
    
    #print info to check
    print(f'method pair {idx_method_pair}: {expl_methods_names[idx_methodA]} ({idx_methodA}), {expl_methods_names[idx_methodB]} ({idx_methodB})')
    print(f'    index of datapoint: {idx_point}')
    print(f'    explanation 1 ({expl_methods_names[idx_methodA]}): {np.around(expl1, 4)}')
    print(f'    explanation 2 ({expl_methods_names[idx_methodB]}): {np.around(expl2, 4)}')
    print(f'    metric values: {np.around(metrics_pointxmetricxpair[idx_point, :, idx_method_pair], 4)}')
    print(f'    metric percentiles: {np.around(ptiles[idx_point, :, idx_method_pair], 4)}')
    
    #create df for plotting
    expl_df = pd.DataFrame({
        'feature': ['age', 'two_year_recid', 'priors_count', 'length_of_stay', 'c_charge_degree', 'sex', 'race'],
        'expl1': expl1,
        'expl2': expl2})

    #plot
    fig, axes = plt.subplots(1, 2, figsize =(16, 4))
    for i in [1, 2]:
        colors = ['cornflowerblue' if x > 0 else 'tomato' for x in expl_df[f'expl{i}']]
        ax = axes[i-1]
        
        #round decimal so that label of bar is rounded
        df = round(expl_df, 2)
        df['expl1'] = np.where(df['expl1'] == -0.00, 0, df['expl1']) #change -0.00 to 0
        df['expl2'] = np.where(df['expl2'] == -0.00, 0, df['expl2'])
        
        #plot
        method1 = sns.barplot(y='feature', x=f'expl{i}', data=df, ax=ax, palette=colors) 
        method_name = expl_methods_names[idx_methodA] if i==1 else expl_methods_names[idx_methodB]
        method1.set(xlabel=f'{method_name} Importance Score', ylabel='Feature', title=method_name)
        
        #make x-axis symmetric around 0
        x_abs_max = abs(max(ax.get_xlim(), key=abs))
        x_abs_max = 1.15*x_abs_max
        ax.set_xlim(xmin=-x_abs_max, xmax=x_abs_max)
        
        #add vertical dotted line at x=0
        ax.axvline(x=0, linestyle='--', color='black')
        
        #add labels for each bar
        for container in ax.containers:
            ax.bar_label(container, padding=5)

    fig.tight_layout()
    plot_path=f'{dataset_name}/figures/user-study/compas_{expl_methods_names[idx_methodA]}_{expl_methods_names[idx_methodB]}_idx{idx_point}.png'
    plt.savefig(plot_path, facecolor='white', transparent=False, dpi=1200)   

