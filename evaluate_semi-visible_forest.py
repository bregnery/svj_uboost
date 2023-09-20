import os, os.path as osp, glob, pickle, logging, warnings, json, math, re
from time import strftime
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

np.random.seed(1001)

from common import logger, DATADIR, Columns, time_and_log, imgcat, set_matplotlib_fontsizes, columns_to_numpy

training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef', 
    # 'phi', 'eta'
    ]
all_features = training_features + ['mt']

# Get the name of all the individual trees
date="Sep09"
decision_trees = []
tree_names  =    ["sig_qcd_mZ200", "sig_qcd_mZ250", "sig_qcd_mZ300", "sig_qcd_mZ350", 
                  "sig_qcd_mZ400", "sig_qcd_mZ450", "sig_qcd_mZ500", "sig_qcd_mZ550", 
                  "sig_qcd_180_to_650", 
                  "sig_tt_mZ200", "sig_tt_mZ250", "sig_tt_mZ300", "sig_tt_mZ350", 
                  "sig_tt_mZ400", "sig_tt_mZ450", "sig_tt_mZ500", "sig_tt_mZ550", 
                  "sig_tt_180_to_650"]
for tree in tree_names :
    decision_trees.append('models/ensemble/bdt_'+date+tree+'.json')

print(decision_trees)

def main():
    set_matplotlib_fontsizes(18, 22, 26)
    qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/QCD_*.npz')]
    qcd_cols = list(filter(lambda c: c.metadata['ptbin'][0]>=300., qcd_cols))
    ttjets_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/TTJets_*.npz')]
    bkg_cols = qcd_cols + ttjets_cols
    #bkg_cols = qcd_cols
    #bkg_cols = ttjets_cols
    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_signal/*.npz')]

    models = {'Semi-visible Forest'    : 'models/ensemble/ensebled_Sep09_semi-visible_forest.json'}
    #          'TT=QCD=1'    : 'models/svjbdt_Sep09_allsignals_qcdttjets.json'}

    plots(signal_cols, bkg_cols, models)


def plots(signal_cols, bkg_cols, models):
    X, y, weight = columns_to_numpy(signal_cols, bkg_cols, features=all_features, downsample=1.)

    import pandas as pd
    X_df = pd.DataFrame(X, columns=all_features)
    mt = X[:,-1]
    X = X[:,:-1]
    # X_eta = X[:,:-1] # Strip off mt
    # X = X_eta[:,:-1] # Also strip off eta

    # _____________________________________________
    # Open the trained models and get the scores

    scores = {}
    aucs = {}

    # xgboost
    import xgboost as xgb

    for key, model_file in models.items():
        if model_file.endswith('.json'):

            # adding an option to evaluate the ensembled network
            if 'semi-visible_forest' in model_file :

                # Make a corresponding model for each tree
                tree_predictions = []
                for tree in decision_trees :
                    tree_model = xgb.XGBClassifier()
                    tree_model.load_model(tree)

                    # Make predicitons with every tree
                    with time_and_log(f'Calculating xgboost scores for {tree}...'):
                        tree_predictions.append(tree_model.predict_proba(X_eta if 'eta' in key else X)[:,1])

                # Combine all preditions into ensemble features
                ensemble_features = np.column_stack(tree_predictions)

                # send predictions to ensembled network for final prediction
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(model_file)
                with time_and_log(f'Calculating xgboost scores for {key}...'):
                    scores[key] = xgb_model.predict_proba(ensemble_features[:,1])
    
            else:
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(model_file)
                with time_and_log(f'Calculating xgboost scores for {key}...'):
                    scores[key] = xgb_model.predict_proba(X_eta if 'eta' in key else X)[:,1]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # uboost
                from hep_ml import uboost
                with open(model_file, 'rb') as f:
                    uboost_model = pickle.load(f)
                    with time_and_log(f'Calculating scores for {key}...'):
                        scores[key] = uboost_model.predict_proba(X_df)[:,1]

    # Sort scores by decreasing auc score
    aucs = {key: roc_auc_score(y, score, sample_weight=weight) for key, score in scores.items()}
    scores = OrderedDict(sorted(scores.items(), key=lambda p: -aucs[p[0]]))
    for key in scores: print(f'{key:50} {aucs[key]}')
    print(aucs)

    # _____________________________________________
    # ROC curves
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    for key, score in scores.items():
        eff_bkg, eff_sig, cuts = roc_curve(y, score, sample_weight=weight)
        ax.plot(
            eff_bkg, eff_sig,
            label=f'{key} (auc={aucs[key]:.3f})'
            )

    if len(scores) <= 10: ax.legend(loc='lower right')
    ax.set_xlabel('bkg eff')
    ax.set_ylabel('sig eff')
    plt.savefig('plots/roc.png', bbox_inches='tight')
    imgcat('plots/roc.png')
    plt.close()

    if len(scores) > 10:
        logger.error('More than 10 models: Not doing individual dist/sculpting plots')
        return


if __name__ == '__main__':
    main()