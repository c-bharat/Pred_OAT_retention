'''
Sept 2020 by Chrianna Bharat
c.bharat@student.unsw.edu.au
Adapted from code by S Barbieri
'''

import sys

sys.path.append('../lib/')

import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch.utils.data as utils
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler

from deep_survival import *
from utils import *
from hyperparameters import Hyperparameters
import optuna
from time import time, ctime

# Objective function that takes three arguments.
def objective(trial, data, df_index_code):
    hp = Hyperparameters(trial)

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

    idx_trn = (data['fold'][:, 0] != 99) # validation IDs are the same across all cols
    uq, cnt = np.unique(idx_trn, return_counts=True)
    print(np.asarray((uq, cnt)).T)

    x_trn_tmp = pd.DataFrame(data['x'][idx_trn], columns=cols_list)
    x_trn = x_mapper.fit_transform(x_trn_tmp)
    time_trn = data['time'][idx_trn]
    event_trn = data['event'][idx_trn]
    codes_trn = data['codes'][idx_trn]
    weeks_trn = data['weeks'][idx_trn]
    diagt_trn = data['diagt'][idx_trn]

    # get index numbers for validation data
    idx_val = (data['fold'][:, 0] == 99)
    uq, cnt = np.unique(idx_val, return_counts=True)
    print(np.asarray((uq, cnt)).T)

    # restrict input variables to those in DataFrameMapper and standardise current fold using parameters from training
    x_val_tmp = pd.DataFrame(data['x'][idx_val], columns=cols_list)
    x_val = x_mapper.transform(x_val_tmp)
    time_val = data['time'][idx_val]
    event_val = data['event'][idx_val]
    codes_val = data['codes'][idx_val]
    weeks_val = data['weeks'][idx_val]
    diagt_val = data['diagt'][idx_val]

    # could move this outside objective function for efficiency
    sort_idx_trn, case_idx_trn, max_idx_control_trn = sort_and_case_indices(x_trn, time_trn, event_trn)
    sort_idx_val, case_idx_val, max_idx_control_val = sort_and_case_indices(x_val, time_val, event_val)

    x_trn, time_trn, event_trn = x_trn[sort_idx_trn], time_trn[sort_idx_trn], event_trn[sort_idx_trn]
    codes_trn, weeks_trn, diagt_trn = codes_trn[sort_idx_trn], weeks_trn[sort_idx_trn], diagt_trn[sort_idx_trn]

    x_val, time_val, event_val = x_val[sort_idx_val], time_val[sort_idx_val], event_val[sort_idx_val]
    codes_val, weeks_val, diagt_val = codes_val[sort_idx_val], weeks_val[sort_idx_val], diagt_val[sort_idx_val]

    # Center continuous variables for IDs not in index list
    p = StandardScaler()
    time_trn = p.fit_transform(time_trn.reshape(-1, 1)).flatten()
    time_val = p.transform(time_val.reshape(-1, 1)).flatten()

    #######################################################################################################

    print('Create data loaders and tensors...')
    case_trn = utils.TensorDataset(torch.from_numpy(x_trn[case_idx_trn]),
                                   torch.from_numpy(time_trn[case_idx_trn]),
                                   torch.from_numpy(max_idx_control_trn),
                                   torch.from_numpy(codes_trn[case_idx_trn]),
                                   torch.from_numpy(weeks_trn[case_idx_trn]),
                                   torch.from_numpy(diagt_trn[case_idx_trn]))
    case_val = utils.TensorDataset(torch.from_numpy(x_val[case_idx_val]),
                                   torch.from_numpy(time_val[case_idx_val]),
                                   torch.from_numpy(max_idx_control_val),
                                   torch.from_numpy(codes_val[case_idx_val]),
                                   torch.from_numpy(weeks_val[case_idx_val]),
                                   torch.from_numpy(diagt_val[case_idx_val]))

    x_trn, x_val = torch.from_numpy(x_trn), torch.from_numpy(x_val)
    time_trn, time_val = torch.from_numpy(time_trn), torch.from_numpy(time_val)
    event_trn, event_val = torch.from_numpy(event_trn), torch.from_numpy(event_val)
    codes_trn, codes_val = torch.from_numpy(codes_trn), torch.from_numpy(codes_val)
    weeks_trn, weeks_val = torch.from_numpy(weeks_trn), torch.from_numpy(weeks_val)
    diagt_trn, diagt_val = torch.from_numpy(diagt_trn), torch.from_numpy(diagt_val)

    # Create batch queues
    trn_loader = utils.DataLoader(case_trn, batch_size=hp.batch_size, shuffle=True, drop_last=True)
    val_loader = utils.DataLoader(case_val, batch_size=hp.batch_size, shuffle=False, drop_last=False)

    print('Train...')
    # Neural Net
    hp.model_name = str(trial.number) + '_' + hp.model_name
    print('Model name is: ', hp.model_name)
    n_inputs = x_trn.shape[1] + 1 if hp.nonprop_hazards else x_trn.shape[1]
    net = NetRNN(n_inputs, df_index_code.shape[0] + 1, hp).to(hp.device)  # +1 for zero padding
    criterion = CoxPHLoss().to(hp.device)
    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

    writer = SummaryWriter(hp.log_dir + "hpo_run/" + str(trial.number))

    best, num_bad_epochs = 100., 0
    for epoch in range(1000):
        trn(trn_loader, x_trn, codes_trn, weeks_trn, diagt_trn, net, criterion, optimizer, hp)
        loss_val = val(val_loader, x_val, codes_val, weeks_val, diagt_val, net, criterion, epoch, hp)
        # early stopping
        if loss_val < best:
            print('############### Saving good model ###############################')
            torch.save(net.state_dict(), hp.log_dir + hp.model_name)
            best = loss_val
            num_bad_epochs = 0
            writer.add_scalar('loss', loss_val, epoch)
        else:
            num_bad_epochs += 1
            if num_bad_epochs == hp.patience:
                break
        # pruning
        trial.report(best, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    print('Done')
    print(ctime())
    return best


def main():
    pp = Hyperparameters()

    print('Load data...')
    data = np.load(pp.data_pp_dir + 'data_arrays.npz')
    print(data['fold'].shape)
    df_index_code = pd.read_feather(pp.data_pp_dir + 'df_index_code.feather')

    # Execute an optimization by using the above objective function wrapped by `lambda`.
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, data, df_index_code), n_trials=100)

    print('Save...')
    save_obj(study, pp.log_dir + 'study.pkl')

if __name__ == '__main__':
    main()