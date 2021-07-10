'''
September 2020 by Chrianna Bharat
c.bharat@student.unsw.edu.au
Adapted from code by: S Barbieri
'''

import sys

sys.path.append('../lib/')

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from deep_survival import *
from hyperparameters import Hyperparameters

from os import listdir
from utils import load_obj

import numpy as np
import pandas as pd
import torch
import pickle as pkl
import lifelines as cph
from sklearn.model_selection import train_test_split

import torch.utils.data as utils
import torch.optim as optim
import torch.nn.functional as F

from pycox.models import CoxCC, CoxTime

from pdb import set_trace as bp
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font', **{'family': 'Times New Roman', 'size': 18})
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.linewidth'] = 1


def list_files1(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def main():
    _ = torch.manual_seed(hp.torch_seed)

    # Load data
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays.npz')
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    df_index_code = pd.read_feather(hp.data_pp_dir + 'df_index_code.feather')

    # Identify covariates to be included in model and subset requiring centering
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    trans_list = []
    for col in cols_list:
        if col in hp.cols_include:
            if col in hp.cols_centre:
                trans_list.append(([col], StandardScaler(with_std=False)))
            else:
                trans_list.append((col, None))
    x_mapper = DataFrameMapper(trans_list)

    print('Test on each fold...')
    for fold in range(5):
        for swap in range(2):

            idx = (data['fold'][:, fold] == swap)
            x_tmp = pd.DataFrame(data['x'][idx], columns=cols_list)
            time = data['time'][idx]
            codes = data['codes'][idx]
            weeks = data['weeks'][idx]
            diagt = data['diagt'][idx]

            # Center continuous variables for IDs not in index list
            x_trn = pd.DataFrame(data['x'][~idx], columns=cols_list)
            x_trn_std = x_mapper.fit(x_trn)
            # Standardise current fold
            x = x_trn_std.transform(x_tmp)
            print(x)

            # Standardise time variable based on training data
            time_trn = data['time'][~idx]
            scaler = StandardScaler().fit(time_trn.reshape(-1, 1))
            time = scaler.transform(time.reshape(-1, 1)).flatten()
            print(time)

            # Convert inputs from numpy array to tensor
            dataset = utils.TensorDataset(torch.from_numpy(x),
                                          torch.from_numpy(time),
                                          torch.from_numpy(codes),
                                          torch.from_numpy(weeks),
                                          torch.from_numpy(diagt))

            # Create batch queues
            loader = utils.DataLoader(dataset, batch_size=hp.batch_size, shuffle=False, drop_last=False)

            # Trained models
            models = [f for f in
                      listdir(hp.log_dir + 'fold_' + str(fold) + '_' + str(1 - swap) + '/')
                      if f.endswith('.pt') & f.startswith('ml_mod_fold_' + str(fold) + '_swp_' + str(1 - swap))]
            print(models)

            # Neural Net
            print(hp.nonprop_hazards, hp.embedding_dim, hp.rnn_type, hp.num_rnn_layers, hp.dropout,
                    hp.num_mlp_layers, hp.add_diagt, hp.add_weeks, hp.summarize, hp.learning_rate)
            n_inputs = x.shape[1]+1 if hp.nonprop_hazards else x.shape[1]
            net = NetRNN(n_inputs, df_index_code.shape[0] + 1, hp)  # +1 for zero padding
            net.eval()

            # Matrix of LPH for all models in dir
            lph_matrix = np.zeros((x.shape[0], len(models)))

            #######################################################################################################

            for i in range(len(models)):

                # Restore variables from disk
                print('Fold: {} Swap: {} Model: {}'.format(fold, swap, i))
                net.load_state_dict(torch.load(hp.log_dir + 'fold_' + str(fold) + '_' + str(1 - swap) + '/'+models[i], map_location='cpu'))
                net.eval()

                # Specify empty array for LPH evaluation
                log_partial_hazard = np.array([])
                with torch.no_grad():
                    for _, (x, time, codes, weeks, diagt) in enumerate(tqdm(loader)):

                        x, time, codes, weeks, diagt = x.to('cpu'), time.to('cpu'), codes.to('cpu'), weeks.to('cpu'), diagt.to('cpu')
                        log_partial_hazard = np.append(log_partial_hazard,
                                                       net(x, codes, weeks, diagt, time=time.unsqueeze(1).float()
                                                           ).detach().cpu().numpy())
                lph_matrix[:, i] = log_partial_hazard

            print('Create dataframe...')
            df_cml = pd.DataFrame(lph_matrix, columns=models)
            print(df_cml.shape)
            df_cml['LPH'] = lph_matrix.mean(axis=1)
            plt.hist(x=df_cml['LPH'], bins='auto', alpha=0.7, rwidth=0.85)
            plt.show()

            print('Saving log partial hazard for fold...')
            df_cml.to_feather(hp.results_dir + 'df_cml_fold_' + str(fold) + '_' + str(swap) + '.feather')

if __name__ == '__main__':
    hp = Hyperparameters()
    main()
