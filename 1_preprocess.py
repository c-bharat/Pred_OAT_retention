'''
Sept 2020 by Chrianna Bharat
c.bharat@student.unsw.edu.au
Adapted from code by: S Barbieri
'''

import pandas as pd
from sas7bdat import SAS7BDAT
from hyperparameters import Hyperparameters as hp

from pdb import set_trace as bp

def main():
    # Load data
    print(hp.priv_data_dir)
    with SAS7BDAT(hp.priv_data_dir + hp.import_dat + '.sas7bdat') as file:
        df = file.to_data_frame()

    # Time to event and binary event column
    df['TIME'] = df['txdays']
    df['EVENT'] = 1-df['cens'] # observed/not-censored
    df['presTenureYears'] = df['presTenureMonths']/12

    # Check length and number of events before saving
    print("Number of rows in 'df'")
    print(df.shape)
    print("Summary of event variable in 'df'")
    print(df.EVENT.value_counts())

    # count the number of nan values in each column
    print("Print counts of variables with any missingness, and remove if needed...")
    na_cnt = df.isnull().sum().to_frame('nulls')
    print(na_cnt[na_cnt['nulls']>0])

    # drop rows with missing values but save/review for reference
    null_data = df[df.isnull().any(axis=1)]
    df.dropna(inplace=True)

    print("Number of rows in 'df'") 
    print(df.shape)
    print("Summary of event variable in 'df'")
    print(df.EVENT.value_counts())

    # Save
    df.reset_index(drop=True, inplace=True)
    print(df.columns)
    df.to_feather(hp.data_pp_dir + hp.import_dat + '_pp.feather')


if __name__ == '__main__':
    main()


