import os
import random
import numpy as np
import pandas as pd
import screen as scr
import pandas as pd
import cum_hazard as ch

if __name__ == '__main__':
    
    # the number of genes in candidate gene I
    num_candidate_I = 100

    save_path = './semi-parametric result'
    data_path = r'./raw data'
    # Protein coding gene
    pc_symbol = pd.read_csv(r'./pc_symbol.csv', index_col=0).gene_name.values.tolist()
    filenames=os.listdir(data_path)
         
    for fl in filenames:
        
        print(fl)
        clinical_data = pd.read_csv(os.path.join(data_path, fl, 'clinical.CSV'), keep_default_na=False)
        exp_data = pd.read_csv(os.path.join(data_path, fl, 'data.csv')); exp_data.index = exp_data['gene_name'].values;
        exp_data = exp_data.loc[np.unique(list(set(exp_data.gene_name.tolist()) & set(pc_symbol))).tolist(), :]

        clinical_data = clinical_data[clinical_data.OS != 0]
        # calculated risk
        clinical_final = ch.cum_hazard(clinical_data)
        
        # screen gene in candidate gene I
        result = scr.screen_step_2(clinical_final, exp_data, h_type = 'hazard_OD', threshold = num_candidate_I)
        
        # save result
        os.makedirs(os.path.join(save_path, fl))
        clinical_final.to_csv(os.path.join(save_path, fl, 'clinical_final.CSV'), sep=',', header=True, index=False)
        result.to_csv(os.path.join(save_path, fl, 'result.csv'),sep=',',index=True,header=True)

