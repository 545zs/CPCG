import os
import whypy
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
from pingouin import partial_corr
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional
from lifelines import KaplanMeierFitter, ExponentialFitter

def skeleton(data, alpha: float):
    n_nodes = data.shape[1]
    labels = data.columns.to_list()
    
    O = [ [[] for _ in range(n_nodes)] for _ in range(n_nodes) ] # 创建分离集列表
    
    G = [ [i != j for i in range(n_nodes)] for j in range(n_nodes) ] # 创建骨架
    
    pairs = [ (i, (n_nodes - j - 1)) for i in range(n_nodes) for j in range(n_nodes - i - 1) ] # 创建点对，不包括自环，是双向边 14=41
    
    done = False
    l = 0  
    while done != True and any(G): 
        done = True
        for x, y in pairs:
            if G[x][y] == True: 
                neighbors = [i for i in range(len(G)) if G[x][i] == True and i != y]  
                if len(neighbors) >= l:
                    done = False
                    for K in set(combinations(neighbors, l)):
                        cc = [ labels[ccc] for ccc in list(K) ]
                        p_value = partial_corr( data=data, x=labels[x], y=labels[y], covar=cc).loc['pearson', 'p-val']
                        if p_value >= alpha:
                            G[x][y] = G[y][x] = False
                            O[x][y] = O[y][x] = list(K)
                            break
        l += 1
    return np.asarray(G, dtype=int), O
 
def cs_step_2(result_cs1, hazard_type):

    data = result_cs1.copy()
    
    labels = data.columns.to_list()
    
    G, O = skeleton(data, alpha = 0.05)
    
    c_idx = np.where(G[:, labels.index(hazard_type)]==1)[0].tolist() + np.where(G[labels.index(hazard_type), :]==1)[0].tolist()

    c_label = list(set([hazard_type] + [labels[idx] for idx in c_idx]))
    c_data = data.loc[:, c_label]
    
    return c_data

if __name__=="__main__":
    # raw data path
    data_path = r'./raw data'
    # parametric data path
    P_data_path = './parametric result'
    # semi-parametric data path
    SP_data_path = './semi-parametric result'
    save_path = r'./result'
    
    cs_filenames=os.listdir(P_data_path)
    fcs_filenames=os.listdir(SP_data_path)
    filenames = list(set(cs_filenames) & set(fcs_filenames))
    
    for fl in filenames:
    
        print(fl)
    
        # Combine candidate gene I and candidate gene II
        P_data = pd.read_csv(os.path.join(P_data_path, fl, 'result.csv'), index_col=0)
        SP_data = pd.read_csv(os.path.join(SP_data_path, fl, 'result.csv'), index_col=0)
        gene_list = list((set(P_data.columns.tolist()) | set(SP_data.columns.tolist())) - set(['hazard_OD', 'OS']))
        
        # get raw data
        clinical_data = pd.read_csv(os.path.join(data_path, fl, 'clinical.CSV'), keep_default_na=False)
        clinical_data.index = clinical_data['case_submitter_id'].values; clinical_data = clinical_data[clinical_data.Censor == 1]
        exp_data = pd.read_csv(os.path.join(data_path, fl, 'data.csv')); exp_data.index = exp_data['gene_name'].values;
        exp_data = exp_data.loc[gene_list, :]; exp_data = exp_data.drop(columns = 'gene_name'); exp_data = exp_data.T
        
        # combine data
        data = pd.merge(clinical_data.OS, exp_data, right_index=True, left_index=True)
        data = data.loc[:, data.corr()['OS'].abs().sort_values(ascending=False).index.tolist()]
        
        # output skeleton
        result = cs_step_2(data, hazard_type = 'OS')
        # save result
        result.to_csv(os.path.join(save_path, '{}.csv'.format(fl)),sep=',',index=True,header=True)
    
