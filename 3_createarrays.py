'''
September 2020 by Chrianna Bharat
c.bharat@student.unsw.edu.au
Adapted from code by: S Barbieri
'''

import pandas as pd
import numpy as np
import pickle as pkl
from hyperparameters import Hyperparameters as hp
from utils import save_obj

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pdb import set_trace as bp

def main():
    np.random.seed(hp.np_seed)

    print('Loading OAT data...')
    df = pd.read_feather(hp.data_pp_dir + hp.import_dat + '_pp.feather')
    print(df.head(10))
    print(df.columns)

    print('Loading DX data...')
    ac = pd.read_feather(hp.data_pp_dir + 'DX_PRES_pp.feather')
    print(ac.head(10))
    print(ac.columns)
    print('-----------------------------------------')

    # numerical index for each person
    df.reset_index(drop=True, inplace=True)
    df_index_person = df['ppn'].reset_index().rename(columns={'index': 'INDEX_PERSON'})

    print('-----------------------------------------')

    # diagnoses
    print('Get max number of codes per person...')
    ac['COUNT'] = ac.groupby(['ppn']).cumcount()
    max_count = ac['COUNT'].max() + 1
    print('max_count {}'.format(max_count))

    code_freq = ac.groupby('CODE').ppn.nunique()
    print(len(code_freq)) #77

    # code index (add 1 to reserve 0 for padding)
    df_index_code = ac[['CODE', 'FORMAT']].drop_duplicates().reset_index(drop=True)
    df_index_code['CODE'] = df_index_code['CODE'].astype(str)
    df_index_code['INDEX_CODE'] = df_index_code.index + 1
    print(df_index_code)

    # codes, times, diag_type arrays
    # (shape: row, shape: col), and desired data type
    codes = np.zeros((len(df_index_person), max_count), dtype=np.int16)  # uint16 not supported by torch
    weeks = np.zeros((len(df_index_person), max_count), dtype=np.uint8)
    diagt = np.zeros((len(df_index_person), max_count), dtype=np.uint8)

    print('Merging index_person...')
    ac = ac.merge(df_index_person, how='left', on='ppn', indicator=True)
    ac = ac[ac._merge != 'left_only'].drop('_merge', axis=1)
    ac['INDEX_PERSON'] = ac['INDEX_PERSON'].astype(int)

    print('Merging index_code...')
    ac['CODE'] = ac['CODE'].astype(str)
    ac = ac.merge(df_index_code, how='left', on=['CODE', 'FORMAT'])
    print(ac.head(20))
    print(ac.dtypes)
    print('Updating arrays...')
    codes[ac['INDEX_PERSON'].values, ac['COUNT'].values] = ac['INDEX_CODE'].values
    weeks[ac['INDEX_PERSON'].values, ac['COUNT'].values] = ac['week_index'].values
    diagt[ac['INDEX_PERSON'].values, ac['COUNT'].values] = ac['DIAG_TYPE'].values

    print('-----------------------------------------')

    # Remove validation data
    df_tmp = df[df.role != 'Validate']

    print('Split data into folds...')
    for i in range(hp.num_folds):
        df_trn, df_tst = train_test_split(df_tmp, test_size=0.5, train_size=0.5, shuffle=True, stratify=df_tmp['EVENT'])
        df['FOLD_' + str(i)] = 99
        df.loc[df_trn.index, 'FOLD_' + str(i)] = 0
        df.loc[df_tst.index, 'FOLD_' + str(i)] = 1
        print('Check fold ' + str(i) + ' frequency values (99s are validation): ')
        print(df.loc[:, 'FOLD_' + str(i)].value_counts())

    # Other arrays
    fold_cols = ['FOLD_' + str(i) for i in range(hp.num_folds)]
    time = df['TIME'].values
    event = df['EVENT'].values.astype(int)
    fold = df[fold_cols].values
    # Specify x array, removing fold vars & variables not included in modelling
    print(len(fold_cols))
    df.drop(hp.cols_base + fold_cols, axis=1, inplace=True)
    x = df.values.astype('float32')
    print(df.shape)
    print(x.shape)
    print(fold.shape)
    for col in df.columns:
        print(col)

    print('-----------------------------------------')
    print('Save...')
    np.savez(hp.data_pp_dir + 'data_arrays.npz', x=x, time=time, event=event, codes=codes, weeks=weeks, diagt=diagt, fold=fold)
    df_index_person.to_feather(hp.data_pp_dir + 'df_index_person.feather')
    df_index_code.to_feather(hp.data_pp_dir + 'df_index_code.feather')
    save_obj(list(df.columns), hp.data_pp_dir + 'cols_list.pkl')

if __name__ == '__main__':
    main()


