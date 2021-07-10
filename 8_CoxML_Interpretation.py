'''
September 2020 by Chrianna Bharat
c.bharat@student.unsw.edu.au
Adapted from code by: S Barbieri
'''

import sys
sys.path.append('../lib/')

from utils import load_obj
from deep_survival import *
from hyperparameters import Hyperparameters
from os import listdir
import statsmodels.stats.api as sms

def main():
    # Load data
    print('Load data...')
    df_index_code = pd.read_feather(hp.data_pp_dir + 'df_index_code.feather')
    icd10_lookup = pd.read_sas(hp.priv_data_dir + 'CODE_ALL_LOOKUP.sas7bdat')
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    means = np.load(hp.data_pp_dir + 'means.npz')
    icd10_lookup[['code', 'description', 'format']] = icd10_lookup[['code', 'description', 'format']].stack().str.decode("utf-8").unstack()

    icd10_lookup = icd10_lookup[['code', 'description', 'format']]
    icd10_lookup.rename(columns={'code': 'CODE', 'description': 'DESCRIPTION', 'format': 'FORMAT'}, inplace=True)
    icd10_lookup['CODE'] = icd10_lookup['CODE'].astype(str)
    icd10_lookup.drop_duplicates(subset='CODE', inplace=True)

    print('Get prevalences and most frequent code type...')
    info_dx = pd.read_feather(hp.data_pp_dir + 'info_dx.feather')
    icd10_lookup = icd10_lookup.merge(info_dx, how='left', on='CODE')

    print('Merge...')
    df_index_code['CODE'] = df_index_code['CODE'].astype(str)
    df_index_code = df_index_code.merge(icd10_lookup, how='left', on=['CODE', 'FORMAT'])
    df_index_code['TYPE'] = 1
    desired_width=320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 10)
    print(df_index_code)
    num_embeddings = df_index_code.shape[0]

    print('Add standard columns...')
    cols_include = []
    for col in cols_list:
        if col in hp.cols_include:
                cols_include.append(col)
    num_cols = len(cols_include)
    df_cols = pd.DataFrame({'TYPE': 0, 'DESCRIPTION': cols_include})
    df_index_code = pd.concat([df_cols, df_index_code], sort=False)
    # reset index
    df_index_code.reset_index(drop=True, inplace=True)

    #######################################################################################################

    print('Compute HRs...')

    # Trained models
    models = listdir(hp.log_dir + 'all/')

    log_hr_columns = np.zeros((num_cols, len(models)))
    log_hr_embeddings = np.zeros((num_embeddings, len(models)))

    # Neural Net
    num_input = num_cols+1 if hp.nonprop_hazards else num_cols
    net = NetRNN(num_input, num_embeddings+1, hp)  # +1 for zero padding
    net.eval()

    for i in range(len(models)):
        print('HRs for model {}'.format(i))

        # Restore variables from disk
        net.load_state_dict(torch.load(hp.log_dir + 'all/' + models[i], map_location='cpu'))

        with torch.no_grad():
            x_b = torch.zeros((1, num_cols), device='cpu')
            times_b = torch.zeros((1, 1), device='cpu')
            codes_b = torch.zeros((1, 1), device='cpu')
            weeks_b = torch.zeros((1, 1), device='cpu')
            diagt_b = torch.zeros((1, 1), device='cpu')
            risk_baseline = net(x_b, codes_b, weeks_b, diagt_b, time=times_b).detach().cpu().numpy().squeeze()

        # Compute risk for standard columns
        for j in tqdm(range(num_cols)):
            with torch.no_grad():
                x_b = torch.zeros((1, num_cols), device='cpu')
                times_b = torch.zeros((1, 1), device='cpu')
                times_b = torch.zeros((1, 1), device='cpu')
                codes_b = torch.zeros((1, 1), device='cpu')
                weeks_b = torch.zeros((1, 1), device='cpu')
                diagt_b = torch.zeros((1, 1), device='cpu')
                x_b[0, j] = 1
                risk_mod = net(x_b, codes_b, weeks_b, diagt_b, time=times_b).detach().cpu().numpy().squeeze() - risk_baseline

            # Store
            log_hr_columns[j, i] = risk_mod

        # Compute risk for embeddings
        for j in tqdm(range(num_embeddings)):
            with torch.no_grad():
                x_b = torch.zeros((1, num_cols), device='cpu')
                times_b = torch.zeros((1, 1), device='cpu')
                codes_b = torch.zeros((1, 1), device='cpu')
                weeks_b = torch.zeros((1, 1), device='cpu')
                diagt_b = torch.zeros((1, 1), device='cpu')
                codes_b[0] = (j+1)
                diagt_b[0] = df_index_code['DIAG_TYPE'].values[num_cols+j]
                risk_mod = net(x_b, codes_b, weeks_b, diagt_b, time=times_b).detach().cpu().numpy().squeeze() - risk_baseline

            # Store
            log_hr_embeddings[j, i] = risk_mod

    # Compute HRs
    log_hr_matrix = np.concatenate((log_hr_columns, log_hr_embeddings))
    mean_hr = np.exp(log_hr_matrix.mean(axis=1))
    lCI, uCI = np.exp(sms.DescrStatsW(log_hr_matrix.transpose()).tconfint_mean())
    df_index_code['HR'] = mean_hr
    df_index_code['lCI'] = lCI
    df_index_code['uCI'] = uCI

    # Save
    df_index_code.to_csv(hp.data_dir + 'DL_hr.csv', index=False)

if __name__ == '__main__':
    hp = Hyperparameters()
    main()
