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
#date="Sep25"
#date="Oct01"
date="Oct05"
old_date="Oct01"
#old_date="Sep13"
directory="models/ensemble/"
#directory="models/weak_ensemble/"
decision_trees = []
old_decision_trees = []
tree_names  =    ["sig_qcd_mZ200", "sig_qcd_mZ250", "sig_qcd_mZ300", "sig_qcd_mZ350", 
                  "sig_qcd_mZ400", "sig_qcd_mZ450", "sig_qcd_mZ500", "sig_qcd_mZ550", 
                  "sig_qcd_180_to_650", 
                  "sig_tt_mZ200", "sig_tt_mZ250", "sig_tt_mZ300", "sig_tt_mZ350", 
                  "sig_tt_mZ400", "sig_tt_mZ450", "sig_tt_mZ500", "sig_tt_mZ550", 
                  "sig_tt_180_to_650",
                  "normal_weights_model"]
for tree in tree_names :
    decision_trees.append(directory+'bdt_'+date+tree+'.json')
for tree in tree_names :
    old_decision_trees.append(directory+'bdt_'+old_date+tree+'.json')

print(decision_trees)

def main():
    set_matplotlib_fontsizes(18, 22, 26)
    qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/QCD_*.npz')]
    qcd_cols = list(filter(lambda c: c.metadata['ptbin'][0]>=300., qcd_cols))
    tt_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/TTJets_*.npz')]
    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_signal/*.npz')]

    models = {#'iterative Semi-visible Forest'    : directory+'ensebled_'+date+'_semi-visible_forest.json',
              'increasing iter Semi-visible Forest'    : directory+'ensebled_'+old_date+'_semi-visible_forest.json',
              #'TT=QCD=1e6 full window'    : 'models/svjbdt_Sep29_allsignals_qcdttjets.json',
              #'TT=QCD=1e6 < 400'    : 'models/svjbdt_Oct24_allsignals_qcdttjets.json',
              'QCD only (iter) < 400'    : 'models/svjbdt_Jan23_lowmass_iterative_qcdonly.json',
              'tt only (iter) < 400'    : 'models/svjbdt_Jan23_lowmass_iterative_ttonly.json',
              #'TT=QCD=1e6 iterative'    : 'models/svjbdt_Sep13_allsignals_iterative_qcdttjets.json',
              #'TT=QCD=1e6 iterative'    : 'models/svjbdt_Sep24_allsignals_iterative_qcdttjets.json',
              #'TT=QCD=1e6 iterative'    : 'models/svjbdt_Oct01_allsignals_iterative_qcdttjets.json',
              #'TT=QCD=1e6 iterative'    : 'models/svjbdt_Oct02_allsignals_iterative_qcdttjets.json',
              'TT=QCD=1e6 many iterations'    : 'models/svjbdt_Nov08_allsignals_iterative_qcdttjets.json',
              'iterative QCD only' : 'models/svjbdt_Jan22_allsignals_iterative_qcdonly.json',
              'iterative tt only' : 'models/svjbdt_Jan22_allsignals_iterative_ttonly.json',
              #'normal full window' : '../models/svjbdt_Aug01_allsignals_qcdttjets.json',
              #'iterative w/ normal weights' : '../models/svjbdt_Aug04_allsignals_iterative_qcdttjets.json',
             }

    # Make ROC curves on the full mT window
    mt_window = [180, 650]

    # Loop over Z' mass windows of +/- 100 GeV 
    mz_prime = [200, 250, 300, 350, 400, 450, 500, 550]
    for mz in mz_prime :

        # define mass window
        mt_window = [mz - 100, mz + 100]
        #mt_window = [180, 650]

        # grab correct signal files
        #signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_signal/*mz' + str(mz) + '*.npz')]
        signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_signal/*mz' + str(mz) + '*rinv0.3' + '*.npz')]

        # make plots using the mt window
        plots(signal_cols, qcd_cols, tt_cols, models, mz, mt_window)


def plots(signal_cols, qcd_cols, tt_cols, models, mz, mt_window):
    all_bkg_cols = [qcd_cols, tt_cols]

    nbkg_loops = 0
    for bkg_cols in all_bkg_cols :

        print("-----------------------------------------------------------")
        if nbkg_loops == 0: print(" QCD Scores vs m(Z')=", str(mz))
        else: print(" tt Scores vs. m(Z')=", str(mz))
        print("-----------------------------------------------------------")

        X, y, weight = columns_to_numpy(signal_cols, bkg_cols, features=all_features, downsample=1., mt_high = mt_window[1], mt_low = mt_window[0])
 
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
                if 'semi-visible_forest' in model_file and date in model_file:
 
                    # Make a corresponding model for each tree
                    tree_predictions = []
                    for tree in decision_trees :
                        tree_model = xgb.XGBClassifier()
                        tree_model.load_model(tree)
 
                        # Make predicitons with every tree
                        #with time_and_log(f'Calculating xgboost scores for {tree}...'):
                        tree_predictions.append(tree_model.predict_proba(X_eta if 'eta' in key else X)[:,1])
 
                    # Combine all preditions into ensemble features
                    ensemble_features = np.column_stack(tree_predictions)
 
                    # send predictions to ensembled network for final prediction
                    xgb_model = xgb.XGBClassifier()
                    xgb_model.load_model(model_file)
                    with time_and_log(f'Calculating xgboost scores for {key}...'):
                        scores[key] = xgb_model.predict_proba(ensemble_features)[:,1]
        
                elif 'semi-visible_forest' in model_file and old_date in model_file :
 
                    # Make a corresponding model for each tree
                    tree_predictions = []
                    for tree in old_decision_trees :
                        tree_model = xgb.XGBClassifier()
                        tree_model.load_model(tree)
 
                        # Make predicitons with every tree
                        #with time_and_log(f'Calculating xgboost scores for {tree}...'):
                        tree_predictions.append(tree_model.predict_proba(X_eta if 'eta' in key else X)[:,1])
 
                    # Combine all preditions into ensemble features
                    ensemble_features = np.column_stack(tree_predictions)
 
                    # send predictions to ensembled network for final prediction
                    xgb_model = xgb.XGBClassifier()
                    xgb_model.load_model(model_file)
                    with time_and_log(f'Calculating xgboost scores for {key}...'):
                        scores[key] = xgb_model.predict_proba(ensemble_features)[:,1]
        
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

        nbkg_loops += 1

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
    if mz != None:
        plt.savefig('plots/mz_' + str(mz) + '_roc.png', bbox_inches='tight')
    else :
        plt.savefig('plots/roc.png', bbox_inches='tight')
    plt.close()

    if len(scores) > 10:
        logger.error('More than 10 models: Not doing individual dist/sculpting plots')
        return


if __name__ == '__main__':
    main()
