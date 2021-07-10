'''
September 2020 by Chrianna Bharat
c.bharat@student.unsw.edu.au
Adapted from code by: S Barbieri
'''

import sys
sys.path.append('../lib/')

from sklearn_pandas import DataFrameMapper
import torch.optim as optim

from deep_survival import *
from hyperparameters import Hyperparameters

from pdb import set_trace as bp
from utils import load_obj
from time import time, ctime
from datetime import datetime
import os

def main():
    _ = torch.manual_seed(hp.torch_seed)

    hp = Hyperparameters()

    # Load data
    print('Load data...')
    data = np.load(hp.data_pp_dir + 'data_arrays.npz')
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    df_index_code = pd.read_feather(hp.data_pp_dir + 'df_index_code.feather')

    # Identify covariates to be included in model and subset requiring centering
    print('Scale continuous features...')
    trans_list = []
    for col in cols_list:
        if col in hp.cols_include:
            if col in hp.cols_centre:
                trans_list.append(([col], StandardScaler(with_std=False)))
            else:
                trans_list.append((col, None))
    x_mapper = DataFrameMapper(trans_list)
    print(x_mapper)

    # Load data excl. validation
    print('Load data...')
    idx = (data['fold'][:, 0] != 99) # validation IDs are the same across all cols
    uq, cnt = np.unique(idx, return_counts=True)
    print(np.asarray((uq, cnt)).T)

    x_tmp = pd.DataFrame(data['x'][idx], columns=cols_list)

    # Save means of non-validation data for interpretation evaluations
    mean_age = x_tmp.loc[:, 'age'].mean()
    mean_presTenureYears = x_tmp.loc[:, 'presTenureYears'].mean()
    np.savez(hp.data_pp_dir + 'means.npz', mean_age=mean_age, mean_presTenureYears=mean_presTenureYears)

    x = x_mapper.fit_transform(x_tmp)
    time = data['time'][idx]
    event = data['event'][idx]
    codes = data['codes'][idx]
    weeks = data['weeks'][idx]
    diagt = data['diagt'][idx]

    sort_idx, case_idx, max_idx_control = sort_and_case_indices(x, time, event)
    x, time, event = x[sort_idx], time[sort_idx], event[sort_idx]
    codes, weeks, diagt = codes[sort_idx], weeks[sort_idx], diagt[sort_idx]

    # Standardise time variable based on training data
    time = StandardScaler().fit_transform(time.reshape(-1, 1)).flatten()

    print('Create data loaders and tensors...')
    case = utils.TensorDataset(torch.from_numpy(x[case_idx]),
                               torch.from_numpy(time[case_idx]),
                               torch.from_numpy(max_idx_control),
                               torch.from_numpy(codes[case_idx]),
                               torch.from_numpy(weeks[case_idx]),
                               torch.from_numpy(diagt[case_idx]))

    x = torch.from_numpy(x)
    time = torch.from_numpy(time)
    event = torch.from_numpy(event)
    codes = torch.from_numpy(codes)
    weeks = torch.from_numpy(weeks)
    diagt = torch.from_numpy(diagt)

    print('HP specs:', hp.nonprop_hazards, hp.embedding_dim, hp.rnn_type, hp.num_rnn_layers, hp.dropout,
          hp.num_mlp_layers, hp.add_diagt, hp.add_weeks, hp.summarize, hp.learning_rate)

    print('Run trials...')
    for trial in range(hp.num_trials):
        print('Trial: {}'.format(trial))

        # Create batch queues
        # Shuffle = True useful for training
        trn_loader = utils.DataLoader(dataset=case, batch_size=hp.batch_size, shuffle=True,  drop_last=True)

        print('Train...', ctime())
        # Neural Net
        mod_name = str(trial) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.pt'
        n_inputs = x.shape[1]+1 if hp.nonprop_hazards else x.shape[1]
        net = NetRNN(n_inputs, df_index_code.shape[0]+1, hp).to(hp.device) #+1 for zero padding
        criterion = CoxPHLoss().to(hp.device)
        optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

        for epoch in range(hp.max_epochs):
            trn(trn_loader, x, codes, weeks, diagt, net, criterion, optimizer, hp)
        os.makedirs(hp.log_dir + 'all/', exist_ok=True)
        torch.save(net.state_dict(), hp.log_dir + 'all/' + mod_name)
        print('Done')

if __name__ == '__main__':
    hp = Hyperparameters()
    main()
