'''
Sept 2020 by Chrianna Bharat
c.bharat@student.unsw.edu.au
'''

import sys
sys.path.append('../lib/')

from hyperparameters import Hyperparameters as hp
from utils import *
from EvalSurv import EvalSurv
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font', **{'family': 'Times New Roman', 'size': 18})
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.linewidth'] = 1

def main():
    data = np.load(hp.data_pp_dir + 'data_arrays.npz')

    time = data['time']
    event = data['event']

    df = pd.DataFrame({'TIME': data['time'], 'EVENT': data['event']})

    # evaluation vectors
    r2d_vec_cox = np.zeros((5, 2))
    din_vec_cox = np.zeros((5, 2))
    con_vec_cox = np.zeros((5, 2))
    ibs_vec_cox = np.zeros((5, 2))
    auc_vec_cox = np.zeros((5, 2))

    r2d_vec_cml = np.zeros((5, 2))
    din_vec_cml = np.zeros((5, 2))
    con_vec_cml = np.zeros((5, 2))
    ibs_vec_cml = np.zeros((5, 2))
    auc_vec_cml = np.zeros((5, 2))

    print('Evaluate on each fold...')
    for fold in range(5):
        for swap in range(2):
            print('Fold: {} Swap: {}'.format(fold, swap))

            idx = (data['fold'][:, fold] == swap)
            df_fold = df[idx].reset_index(drop=True)
            print(df_fold.shape)

            df_cox = df_fold.copy()
            df_cml = df_fold.copy()

            # load log partial hazards
            df_cox['LPH'] = pd.read_feather(hp.results_dir + 'df_cox_fold_' + str(fold) + '_' + str(swap) + '.feather')
            df_cml['LPH'] = pd.read_feather(hp.results_dir + 'df_cml_fold_' + str(fold) + '_' + str(swap) + '.feather')['LPH']

            ################################################################################################

            es_cox = EvalSurv(df_cox.copy())
            es_cml = EvalSurv(df_cml.copy())

            r2d_vec_cox[fold, swap] = es_cox.R_squared_D()
            din_vec_cox[fold, swap], _ = es_cox.D_index()
            con_vec_cox[fold, swap] = es_cox.concordance_index()
            ibs_vec_cox[fold, swap] = es_cox.integrated_brier_score(hp.times)
            auc_vec_cox[fold, swap] = es_cox.auc(hp.times)

            r2d_vec_cml[fold, swap] = es_cml.R_squared_D()
            din_vec_cml[fold, swap], _ = es_cml.D_index()
            con_vec_cml[fold, swap] = es_cml.concordance_index()
            ibs_vec_cml[fold, swap] = es_cml.integrated_brier_score(hp.times)
            auc_vec_cml[fold, swap] = es_cml.auc(hp.times)

    print('R-squared(D) ')
    print(r2d_vec_cox)
    print(r2d_vec_cml)

    print('D-index')
    print(din_vec_cox)
    print(din_vec_cml)

    print('Concordance')
    print(con_vec_cox)
    print(con_vec_cml)

    print('IBS')
    print(ibs_vec_cox)
    print(ibs_vec_cml)

    # 5x2 cv F Test for Comparing Supervised Classification Learning Algorithms
    r2d_p = robust_cv_test(r2d_vec_cox, r2d_vec_cml)
    print('R-squared(D) p-value: {:.3}'.format(r2d_p))
    r2d_vec_cox = np.reshape(r2d_vec_cox, -1)
    r2d_mean, (r2d_lci, r2d_uci) = r2d_vec_cox.mean(), sms.DescrStatsW(r2d_vec_cox).tconfint_mean()
    print('R-squared(D) Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(r2d_mean, r2d_lci, r2d_uci))
    r2d_vec_cml = np.reshape(r2d_vec_cml, -1)
    r2_mean, (r2_lci, r2_uci) = r2d_vec_cml.mean(), sms.DescrStatsW(r2d_vec_cml).tconfint_mean()
    print('R-squared(D) CML (95% CI): {:.3} ({:.3}, {:.3})'.format(r2_mean, r2_lci, r2_uci))
    din_p = robust_cv_test(din_vec_cox, din_vec_cml)
    print('D-index p-value: {:.3}'.format(din_p))
    con_p = robust_cv_test(con_vec_cox, con_vec_cml)
    print('Concordance p-value: {:.3}'.format(con_p))
    ibs_p = robust_cv_test(ibs_vec_cox, ibs_vec_cml)
    print('IBS p-value: {:.3}'.format(ibs_p))
    auc_p = robust_cv_test(auc_vec_cox, auc_vec_cml)
    print('AUC p-value: {:.3}'.format(auc_p))

    # When using a -1, the dimension corresponding to the -1 will be the product of the dimensions of the original array
    # divided by the product of the dimensions given to reshape so as to maintain the same number of elements.
    r2d_vec_cox = np.reshape(r2d_vec_cox, -1)
    din_vec_cox = np.reshape(din_vec_cox, -1)
    con_vec_cox = np.reshape(con_vec_cox, -1)
    ibs_vec_cox = np.reshape(ibs_vec_cox, -1)
    auc_vec_cox = np.reshape(auc_vec_cox, -1)
    r2d_vec_cml = np.reshape(r2d_vec_cml, -1)
    din_vec_cml = np.reshape(din_vec_cml, -1)
    con_vec_cml = np.reshape(con_vec_cml, -1)
    ibs_vec_cml = np.reshape(ibs_vec_cml, -1)
    auc_vec_cml = np.reshape(auc_vec_cml, -1)

    r2d_mean, (r2d_lci, r2d_uci) = r2d_vec_cox.mean(), sms.DescrStatsW(r2d_vec_cox).tconfint_mean()
    print('R-squared(D) Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(r2d_mean, r2d_lci, r2d_uci))
    din_mean, (din_lci, din_uci) = din_vec_cox.mean(), sms.DescrStatsW(din_vec_cox).tconfint_mean()
    print('D-index Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(din_mean, din_lci, din_uci))
    con_mean, (con_lci, con_uci) = con_vec_cox.mean(), sms.DescrStatsW(con_vec_cox).tconfint_mean()
    print('Concordance Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(con_mean, con_lci, con_uci))
    ibs_mean, (ibs_lci, ibs_uci) = ibs_vec_cox.mean(), sms.DescrStatsW(ibs_vec_cox).tconfint_mean()
    print('IBS Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(ibs_mean, ibs_lci, ibs_uci))
    auc_mean, (auc_lci, auc_uci) = auc_vec_cox.mean(), sms.DescrStatsW(auc_vec_cox).tconfint_mean()
    print('AUC Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(auc_mean, auc_lci, auc_uci))

    r2_mean, (r2_lci, r2_uci) = r2d_vec_cml.mean(), sms.DescrStatsW(r2d_vec_cml).tconfint_mean()
    print('R-squared(D) CML (95% CI): {:.3} ({:.3}, {:.3})'.format(r2_mean, r2_lci, r2_uci))
    din_mean, (din_lci, din_uci) = din_vec_cml.mean(), sms.DescrStatsW(din_vec_cml).tconfint_mean()
    print('D-index CML (95% CI): {:.3} ({:.3}, {:.3})'.format(din_mean, din_lci, din_uci))
    con_mean, (con_lci, con_uci) = con_vec_cml.mean(), sms.DescrStatsW(con_vec_cml).tconfint_mean()
    print('Concordance CML (95% CI): {:.3} ({:.3}, {:.3})'.format(con_mean, con_lci, con_uci))
    ibs_mean, (ibs_lci, ibs_uci) = ibs_vec_cml.mean(), sms.DescrStatsW(ibs_vec_cml).tconfint_mean()
    print('IBS CML (95% CI): {:.3} ({:.3}, {:.3})'.format(ibs_mean, ibs_lci, ibs_uci))
    auc_mean, (auc_lci, auc_uci) = auc_vec_cml.mean(), sms.DescrStatsW(auc_vec_cml).tconfint_mean()
    print('AUC CML (95% CI): {:.3} ({:.3}, {:.3})'.format(auc_mean, auc_lci, auc_uci))

    print('Save...')
    np.savez(hp.data_dir + 'eval_vecs.npz',
             r2d_vec_cox=r2d_vec_cox, din_vec_cox=din_vec_cox, con_vec_cox=con_vec_cox, ibs_vec_cox=ibs_vec_cox,
             auc_vec_cox=auc_vec_cox,
             r2d_vec_cml=r2d_vec_cml, din_vec_cml=din_vec_cml, con_vec_cml=con_vec_cml, ibs_vec_cml=ibs_vec_cml,
             auc_vec_cml=auc_vec_cml,
             r2d_p=r2d_p, din_p=din_p, con_p=con_p, ibs_p=ibs_p, auc_p=auc_p
             )

if __name__ == '__main__':
    main()