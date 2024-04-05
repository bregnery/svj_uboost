#==============================================================================
# apply_DDT.py ----------------------------------------------------------------
#------------------------------------------------------------------------------
# Author(s): Brendan Regnery, Sara Nabili -------------------------------------
#------------------------------------------------------------------------------
# Applies a DDT to a trained BDT model ----------------------------------------
#    (Designed Decorrelated Tagger, https://arxiv.org/pdf/1603.00027.pdf) -----
#------------------------------------------------------------------------------

import os, os.path as osp, glob, pickle, logging, warnings, json, math, re
from time import strftime
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter


np.random.seed(1001)

from common import logger, DATADIR, Columns, time_and_log, imgcat, set_matplotlib_fontsizes, columns_to_numpy, columns_to_numpy_single


training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef', 
    # 'phi', 'eta'
    ]
all_features = training_features + ['pt', 'mt', 'rho'] # rho is an important variable for applying the decorrelation

model_file = 'models/svjbdt_Feb28_lowmass_iterative_qcdtt_100p38.json'

#------------------------------------------------------------------------------
# Program specific function ---------------------------------------------------
#------------------------------------------------------------------------------

# Sets an optimal window for applying the DDT
def rhoddt_windowcuts(mt, pt, rho):
    cuts = (pt>110) & (pt<1500) & (rho>-4) & (rho<0)
    return cuts

# This is the variable map of the BDT discriminator
def varmap(mt, pt, rho, var, weight, percent):
    cuts = rhoddt_windowcuts(mt, pt, rho)
    C, RHO_edges, PT_edges = np.histogram2d(rho[cuts], pt[cuts], bins=49,weights=weight[cuts])
    w, h = 50, 50
    VAR_map      = [[0 for x in range(w)] for y in range(h)]
    VAR = var[cuts]
    for i in range(len(RHO_edges)-1):
       for j in range(len(PT_edges)-1):
          CUT = (rho[cuts]>RHO_edges[i]) & (rho[cuts]<RHO_edges[i+1]) & (pt[cuts]>PT_edges[j]) & (pt[cuts]<PT_edges[j+1])
          if len(VAR[CUT])==0: continue
          if len(VAR[CUT])>0:
             #VAR_map[i][j]=np.percentile(VAR[CUT],18.2) # bdt>0.6
             VAR_map[i][j]=np.percentile(VAR[CUT],100-percent) # bdt>0.4

    VAR_map_smooth = gaussian_filter(VAR_map,1)
    return VAR_map_smooth, RHO_edges, PT_edges

# This is the actual DDT function which returns the DDT corrected version of the BDT discriminator
def ddt(mt, pt, rho, var_map, var, weight, percent):
    with time_and_log(f'Calculating ddt scores for ...{percent}'):
        cuts = rhoddt_windowcuts(mt, pt, rho)
        var_map_smooth, RHO_edges, PT_edges = varmap(mt, pt, rho, var_map, weight, percent)
        nbins = 49
        Pt_min, Pt_max = min(PT_edges), max(PT_edges)
        Rho_min, Rho_max = min(RHO_edges), max(RHO_edges)

        ptbin_float  = nbins*(pt-Pt_min)/(Pt_max-Pt_min)
        rhobin_float = nbins*(rho-Rho_min)/(Rho_max-Rho_min)

        # Not sure of the differences between the two, must ask Sara
        '''ptbin  = np.clip(1 + np.round(ptbin_float).astype(int), 0, nbins)
        rhobin = np.clip(1 + np.round(rhobin_float).astype(int), 0, nbins)'''
        ptbin  = np.clip(1 + ptbin_float.astype(int), 0, nbins)
        rhobin = np.clip(1 + rhobin_float.astype(int),0, nbins)

        varDDT = np.array([var[i] - var_map_smooth[rhobin[i]-1][ptbin[i]-1] for i in range(len(var))])
        return varDDT
        #return varDDT, rhobin, ptbin, var_map_smooth, RHO_edges, PT_edges

#------------------------------------------------------------------------------
# The Main Function -----------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    set_matplotlib_fontsizes(18, 22, 26)

    # Grab the testing data
    qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/QCD_*.npz')]
    qcd_cols = list(filter(lambda c: c.metadata['ptbin'][0]>=300., qcd_cols))
    ttjets_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/TTJets_*.npz')]
    bkg_cols = qcd_cols + ttjets_cols
    #bkg_cols = qcd_cols
    #bkg_cols = ttjets_cols
    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_signal/*.npz')]

    #X, y, weight = columns_to_numpy(signal_cols, bkg_cols, features=all_features, downsample=1.)
    X, bkg_weight = columns_to_numpy_single(bkg_cols, features=all_features)

    X_df = pd.DataFrame(X, columns=all_features)

    # grab rho
    rho = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # grab mT
    mT = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # grab pT
    pT = X[:,-1]
    X = X[:,:-1] # remove it from X so that eventually it can be used for BDT scores

    # _____________________________________________
    # Open the trained models and get the scores

    bkg_score = {}

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_file)
    with time_and_log(f'Calculating xgboost scores for {model_file}...'):
        bkg_score = xgb_model.predict_proba(X)[:,1]

    # _____________________________________________
    # Calculate bkg efficiencies for the DDT

    bkg_eff=[]
    bkg_Hist={}
    for i in range(0,10):
        bkg_Hist[i]=np.histogram(bkg_score[bkg_score>i/10],weights=bkg_weight[bkg_score>i/10]*len(bkg_score))
        bkg_eff.append(sum(bkg_Hist[i][0])/sum(bkg_Hist[0][0]))

    # _____________________________________________
    # Apply the DDT 

    BKG_score_ddt = []
    for cuts in (-1,0,1,2,3,4,5):
        index = int(cuts) + 1
        BKG_score_ddt.append(ddt(mT, pT, rho, bkg_score, bkg_score, bkg_weight, bkg_eff[index]*100) )

    print(BKG_score_ddt) 

    



if __name__ == '__main__':
    main()
