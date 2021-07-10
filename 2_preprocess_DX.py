'''
September 2020 by Chrianna Bharat
c.bharat@student.unsw.edu.au
Adapted from code by: S Barbieri
'''

import pandas as pd
from sas7bdat import SAS7BDAT
from hyperparameters import Hyperparameters as hp

from pdb import set_trace as bp


def main():
    df = pd.DataFrame()
    df = pd.read_sas(hp.priv_data_dir + 'DX_PRES_LONG.sas7bdat')
    print(df.head())
    df.rename(columns={'dx1': 'CODE'}, inplace=True)
    df['week_index'] = df['week_index'].astype(int)
    df.rename(columns={'format': 'FORMAT'}, inplace=True)
    df[['CODE', 'ppn', 'FORMAT', 'data_source', 'pos']] = df[['CODE', 'ppn', 'FORMAT', 'data_source', 'pos']].stack().str.decode("utf-8").unstack()
    print(df.head())
    print(pd.crosstab(index=df["data_source"], columns=df["FORMAT"]))

    print('Remove future data...')
    df = df[df['week_index'] <= 53]

    print('Remove codes associated with less than min_count persons...')
    df = df[df.groupby(['CODE', 'FORMAT'])['ppn'].transform('nunique') >= hp.min_count]
    codes = pd.DataFrame(df.groupby(['CODE', 'FORMAT'])['ppn'].nunique())
    print(len(codes))
    print(codes.to_string())
    print(codes.shape)

    print('Replace DIAG_TYP with numerical values...')
    df.rename(columns={'pos': 'DIAG_TYPE'}, inplace=True)
    df['DIAG_TYPE'] = df['DIAG_TYPE'].replace({'P': 1, 'S': 2})
    print(df.DIAG_TYPE.value_counts(dropna=False))

    print('Code prevalence and most frequent diag type...')
    info_dx = df.groupby(['CODE'])[['ppn', 'DIAG_TYPE']]
    info_dx = info_dx.agg({'ppn': lambda x: x.nunique(), 'DIAG_TYPE': lambda x: pd.Series.mode(x)[0]}).reset_index()
    info_dx.rename(columns={'ppn': 'PREVALENCE'}, inplace=True)
    print(info_dx)

    print('Save...')
    info_dx.to_feather(hp.data_pp_dir + 'info_dx.feather')

    print('Save...')
    df.sort_values(by=['ppn', 'week_index', 'CODE', 'FORMAT'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_feather(hp.data_pp_dir + 'DX_PRES_pp.feather')

if __name__ == '__main__':
    main()

