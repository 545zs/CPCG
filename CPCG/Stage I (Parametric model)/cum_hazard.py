import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import expon, chisquare, kstest
from lifelines import KaplanMeierFitter, ExponentialFitter

def cum_hazard(clinical_data):
    
    cd = clinical_data.copy()
    
    c_dead_OD = cd[cd['vital_status'] == 'Dead']
    kmf_OD = KaplanMeierFitter()
    kmf_OD.fit(c_dead_OD['OS'], c_dead_OD['Censor'])
    event_table_OD = kmf_OD.event_table
    event_table_OD['hazard_OD'] = event_table_OD['observed'] / event_table_OD['at_risk']
    sur_fun_OD = kmf_OD.survival_function_
    sur_fun_OD['hazard_OD'] = event_table_OD['hazard_OD']
    sur_fun_OD = sur_fun_OD.drop('KM_estimate', axis=1)

    result = pd.merge(cd, sur_fun_OD, how='left', left_on='OS', right_index=True).copy()

    result['Censor_OD'] = result['Censor']
    data_OD = result[result['Censor_OD'] == 1]['hazard_OD']
    scale = np.mean(data_OD); em_OD = expon(scale=scale);
    # exclude a minimal number of samples with outlier survival time
    for per in range(len(data_OD)):
        threshold_OD = np.percentile(em_OD.pdf(data_OD), per+1)
        filtered_sample_OD = [x for x in data_OD if em_OD.pdf(x) >= threshold_OD]
        
        params = stats.expon.fit(filtered_sample_OD); loc, scale = params;
        _, p_value = kstest(filtered_sample_OD, 'expon', args=(loc, scale))
        if p_value > 0.1:
            break
    for index in range(result.shape[0]):
        if result.iloc[index, result.columns.get_loc('hazard_OD')] not in filtered_sample_OD:
            result.iloc[index, result.columns.get_loc('Censor_OD')] = 0
    
    result['hazard_OD'] = np.log(result['hazard_OD'])
    result.loc[result.Censor_OD==0, ['hazard_OD']] = 999
    
    return result



