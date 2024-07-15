import os
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, ExponentialFitter

def cum_hazard(clinical_data):
    
    cd = clinical_data.copy()
    
    c_dead_OD = cd.copy()
    kmf_OD = KaplanMeierFitter()
    kmf_OD.fit(c_dead_OD['OS'], c_dead_OD['Censor'])
    event_table_OD = kmf_OD.event_table
    event_table_OD['hazard_OD'] = event_table_OD['observed'] / event_table_OD['at_risk']
    sur_fun_OD = kmf_OD.survival_function_
    sur_fun_OD['hazard_OD'] = event_table_OD['hazard_OD']
    sur_fun_OD = sur_fun_OD.drop('KM_estimate', axis=1)
    
    result = pd.merge(cd, sur_fun_OD, how='left', left_on='OS', right_index=True).copy()
    
    result.loc[result.Censor==0, ['hazard_OD']] = 999
    
    return result