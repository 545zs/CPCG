import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
from pingouin import partial_corr
import statsmodels.api as sm
from lifelines.statistics import logrank_test

np.seterr(divide='ignore',invalid='ignore')

def screen_step_2(clinical_final, exp_data, h_type, threshold = 100):
    
    cd = clinical_final.copy()
    ed = exp_data.copy()
    
    cd.index = cd['case_submitter_id'].values 
    
    table = pd.DataFrame(index = exp_data.gene_name.tolist(), columns = ['corr', 'logrank'])
    
    for aa in range(ed.shape[0]): 
        
        temp_data = ed[aa:aa+1].T.copy()
        name_gene = temp_data.loc['gene_name'].values # 设置基因名
        temp_data.columns = name_gene # 设置列名
        temp_data = temp_data.drop(['gene_name'])
        
        cd = cd.merge(temp_data, how='left', left_index=True, right_index=True)
        cd[name_gene[0]] = cd[name_gene[0]].astype(float)
        
        # 计算与OS的相关性
        d_l_mean = cd[cd[name_gene[0]] < cd[name_gene[0]].mean()].copy();
        d_h_mean = cd[cd[name_gene[0]] > cd[name_gene[0]].mean()].copy();
        # Logrank test
        results = logrank_test(d_l_mean['OS'], d_h_mean['OS'], d_l_mean['Censor'], d_h_mean['Censor'])

        if results.p_value > 0.01:
            cd = cd.drop(columns=name_gene[0])
            continue
        table.loc[name_gene[0], 'logrank'] = results.p_value/2
        
        corr_pd = partial_corr(data=cd[cd['Censor']==1], x=name_gene[0], y=h_type).loc['pearson','r']
        table.loc[name_gene[0], 'corr'] = np.abs(corr_pd)
            
        cd = cd.drop(columns=name_gene[0])
    
    table = table.dropna(axis=0,how='all')
    table[['corr']] = table[['corr']].astype(float); table[['logrank']] = table[['logrank']].astype(float);
    
    if table.shape[0] < threshold:
        print('table.shape[0] < threshold')
        threshold = table.shape[0]
    corr_index = table.sort_values(by = 'corr', ascending=False).iloc[0:threshold,:].index.tolist()
    
    ed.index = ed['gene_name'].values
    
    result = pd.DataFrame(); result.index = cd[cd['Censor']==1].index
    result = pd.merge(result, cd[cd['Censor']==1][[h_type, 'OS']], how='left', left_index=True, right_index=True) # h_type
    result = pd.merge(result, ed.loc[corr_index, :].drop(columns = 'gene_name').T, how='left', left_index=True, right_index=True) # h_type

    return result
