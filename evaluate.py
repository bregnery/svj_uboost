import os, os.path as osp, glob, pickle, logging, warnings, json, math
from time import strftime
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

np.random.seed(1001)

from common import logger, DATADIR, Columns, time_and_log, imgcat, set_matplotlib_fontsizes, columns_to_numpy


training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi', 'phi'
    ]
all_features = training_features + ['mt']


def main():
    set_matplotlib_fontsizes(18, 22, 26)

    qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/QCD_*.npz')]
    qcd_cols = list(filter(lambda c: c.metadata['ptbin'][0]>=300., qcd_cols))
    ttjets_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/TTJets_*.npz')]
    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_signal/*.npz')]

    X, y, weight = columns_to_numpy(signal_cols, qcd_cols+ttjets_cols, features=all_features, downsample=1.)

    import pandas as pd
    X_df = pd.DataFrame(X, columns=all_features)
    mt = X[:,-1]
    X = X[:,:-1]

    # _____________________________________________
    # Open the trained models and get the scores

    scores = OrderedDict()

    # uboost
    from hep_ml import uboost

    with open('model_uboost_Nov11.pkl', 'rb') as f:
        uboost_model = pickle.load(f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with time_and_log('Calculating uboost scores...'):
            scores['uboost'] = uboost_model.predict_proba(X_df)[:,1]

    # xgboost
    import xgboost as xgb

    xgb_models = {
        'unreweighted' : 'models/svjbdt_Nov10.json',
        'girth_reweighted' : 'models/svjbdt_Nov11_reweight_girth.json',
        'mt_reweighted' : 'models/svjbdt_Nov11_reweight_mt.json',
        'pt_reweighted' : 'models/svjbdt_Nov11_reweight_pt.json',
        }
    
    for key, model_file in xgb_models.items():
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(model_file)
        with time_and_log(f'Calculating xgboost scores for {key}...'):
            scores[key] = xgb_model.predict_proba(X)[:,1]
            print(scores[key][:5])


    # _____________________________________________
    # ROC curves
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    for key, score in scores.items():
        eff_bkg, eff_sig, cuts = roc_curve(y, score, sample_weight=weight)
        auc = roc_auc_score(y, score, sample_weight=weight)
        ax.plot(
            eff_bkg, eff_sig,
            label=f'{key} (auc={auc:.2f})'
            )

    ax.legend()
    ax.set_xlabel('bkg eff')
    ax.set_ylabel('sig eff')
    plt.savefig('plots/roc.png', bbox_inches='tight')
    imgcat('plots/roc.png')
    plt.close()

    # _____________________________________________
    # Score distributions

    n_cols = 2
    n_rows = math.ceil(len(scores)/n_cols)
    fig = plt.figure(figsize=(8*n_cols, 8*n_rows))

    for i, (key, score) in enumerate(scores.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.set_title(key)

        bins = np.linspace(0, 1, 40)
        if key=='uboost':
            bins = np.linspace(min(score), max(score), 40)

        ax.hist(score[y==0], bins, density=True, label='Bkg')
        ax.hist(score[y==1], bins, density=True, label='Signal', alpha=.6)

        ax.legend()
        ax.set_xlabel('BDT Score')
        ax.set_ylabel('A.U.')

    plt.savefig(f'plots/scorehist.png', bbox_inches='tight')
    imgcat(f'plots/scorehist.png')
    plt.close()


    # _____________________________________________
    # Bkg sculpting check

    n_cols = 2
    n_rows = len(scores)
    fig = plt.figure(figsize=(16,8*n_rows))
    mt_bkg = mt[y==0]

    for i, (key, score) in enumerate(scores.items()):
        score_bkg = score[y==0]

        for density in [True, False]:
            ax = fig.add_subplot(n_rows, n_cols, 2*i+1+density)
            ax.set_title(key + (' (normed)' if density else ''), fontsize=28)
            bins = np.linspace(0, 800, 80)

            cuts = np.linspace(.0, .9, 10)
            if key=='uboost': cuts = np.linspace(min(score_bkg), max(score_bkg), 11)[:-1]

            for cut in cuts:
                ax.hist(mt_bkg[score_bkg>cut], bins, histtype='step', label=f'score>{cut:.2f}', density=density)

            ax.legend()
            ax.set_xlabel('mT (GeV)')
            ax.set_ylabel('A.U.')
        
    outfile = f'plots/mthist.png'
    plt.savefig(outfile, bbox_inches='tight')
    imgcat(outfile)
    plt.close()


if __name__ == '__main__':
    main()