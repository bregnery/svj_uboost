#=========================================================================================
# plot_input_features.py -----------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Plot some input features for debugging purposes ----------------------------------------
#-----------------------------------------------------------------------------------------

import os, os.path as osp, glob, pickle, logging, argparse, sys, re, pprint
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg') # prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS") # CMS plot style
from common import logger, DATADIR, Columns, time_and_log, columns_to_numpy, set_matplotlib_fontsizes, imgcat, add_key_value_to_json, filter_pt

# Load a BDT with xgboost
model_file = '../models/svjbdt_Aug01_allsignals_qcdttjets.json'
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(model_file)

# Define a list of .npz filenames
tt_cols = [Columns.load(f) for f in glob.glob(DATADIR + '/train_bkg/Summer20UL18/TTJets_*.npz')]
qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR + '/train_bkg/Summer20UL18/QCD_*.npz')]
sig_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*.npz')]

# BDT features
bdt_features = [
        'girth', 'ptd', 'axismajor', 'axisminor',
        'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
        'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef'
        ]

# Features to plot
features = [
    ['weight'], ['puweight'], ['girth'], ['ptd'], ['axismajor'], ['axisminor'],
    ['ecfm2b1'], ['ecfd2b1'], ['ecfc2b1'], ['ecfn2b2'], ['metdphi'],
    ['ak15_chad_ef'], ['ak15_nhad_ef'], ['ak15_elect_ef'], ['ak15_muon_ef'], ['ak15_photon_ef'], 
    ]

# Initialize an empty dictionary to store feature data
tt_data = {}
qcd_data = {}
sig_data = {}

# weighting normalization
k = 1000000.0

# Load feature data from each .npz file and store it in the dictionary
tt_weight = []
qcd_weight = []
sig_weight = []
for feature in features :
    for cols in qcd_cols:
        this_bkg = cols.to_numpy(feature)
        feature_name = feature[0]

        if feature_name not in qcd_data:
            qcd_data[feature_name] = []
        if feature_name == 'weight' :
            this_weight = cols.arrays['puweight']*cols.arrays['weight']
            qcd_weight.append(this_weight)
            #len_qcd_cols = len(this_bkg)
            #qcd_weight.append((1./len_qcd_cols)*np.ones(len_qcd_cols))
        qcd_data[feature_name].append(this_bkg)

# apply reweighting normalization
qcd_weight = np.concatenate(qcd_weight)
total_weight = sum(qcd_weight)
qcd_weight = [weight * (k / total_weight) for weight in qcd_weight]
qcd_weight = np.array(qcd_weight)

for feature in features :
    for cols in tt_cols:
        this_bkg = cols.to_numpy(feature)
        feature_name = feature[0]

        if feature_name not in tt_data:
            tt_data[feature_name] = []
        if feature_name == 'weight' :
            this_weight = cols.arrays['puweight']*cols.arrays['weight']
            tt_weight.append(this_weight)
            #len_tt_cols = len(this_bkg)
            #tt_weight.append((1./len_tt_cols)*np.ones(len_tt_cols))
        tt_data[feature_name].append(this_bkg)

# apply reweighting normalization
tt_weight = np.concatenate(tt_weight)
# Set total signal weight equal to total bkg weight
tt_weight *= np.sum(qcd_weight) / np.sum(tt_weight)
#tt_weight = [weight * (k / total_weight) for weight in tt_weight]
tt_weight = np.array(tt_weight)

# For the SVJ Signal
for feature in features :
    for cols in sig_cols:
        this_sig = cols.to_numpy(feature)
        feature_name = feature[0]

        if feature_name not in sig_data:
            sig_data[feature_name] = []
        if feature_name == 'weight' :
            len_sig_cols = len(this_sig)
            sig_weight.append((1./len_sig_cols)*np.ones(len_sig_cols))
        sig_data[feature_name].append(this_sig)

# apply reweighting normalization
sig_weight = np.concatenate(sig_weight)
# Set total signal weight equal to total bkg weight
sig_weight *= np.sum(tt_weight) / np.sum(sig_weight)
sig_weight = np.array(sig_weight)


# store BDT input features
tt_X = [] 
qcd_X = [] 
sig_X = [] 
for cols in qcd_cols:
    this_bkg = cols.to_numpy(bdt_features)
    qcd_X.append(this_bkg)
for cols in tt_cols:
    this_bkg = cols.to_numpy(bdt_features)
    tt_X.append(this_bkg)
for cols in sig_cols:
    this_sig = cols.to_numpy(bdt_features)
    sig_X.append(this_sig)
tt_X = np.concatenate(tt_X)
qcd_X = np.concatenate(qcd_X)
sig_X = np.concatenate(sig_X)

# Convert lists to numpy arrays
for key in qcd_data.keys():
    qcd_data[key] = np.concatenate(qcd_data[key])
for key in tt_data.keys():
    tt_data[key] = np.concatenate(tt_data[key])
for key in sig_data.keys():
    sig_data[key] = np.concatenate(sig_data[key])

# BDT cut values
bdt_cuts = [0.1, 0.4, 0.8]

# Get the BDT probabilities
prob_tt = xgb_model.predict_proba(tt_X)[:,1]
prob_qcd = xgb_model.predict_proba(qcd_X)[:,1]
prob_sig = xgb_model.predict_proba(sig_X)[:,1]

# Plot features for different BDT cut values
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
for key in tt_data.keys():

    # set bin size before looping over BDT cuts
    data_range = (min(np.min(tt_data[key]), np.min(qcd_data[key]), np.min(sig_data[key])), max(np.max(tt_data[key]), np.max(qcd_data[key]), np.max(sig_data[key])))

    nloops = 0
    for bdt_cut in bdt_cuts: 

        # Get the BDT cut mask
        bdt_tt_mask = prob_tt > bdt_cut
        bdt_qcd_mask = prob_qcd > bdt_cut
        bdt_sig_mask = prob_sig > bdt_cut

        # Calculate the bin edges from the data range
        n_bins = 50
        bins = np.linspace(data_range[0], data_range[1], n_bins)

        # make darker color every time
        alpha = 0.2 + nloops*0.4

        # Plot the histograms
        hep.cms.label(rlabel="Work in Progress") # Options to make the plot fancier 
        # high light the highest BDT cut
        if nloops == 2:
            plt.hist(sig_data[key][bdt_sig_mask], bins=bins, weights=sig_weight[bdt_sig_mask], histtype='stepfilled', color='orange', label=f'Signal > {str(bdt_cut)}')
            plt.hist(tt_data[key][bdt_tt_mask], bins=bins, weights=tt_weight[bdt_tt_mask], histtype='stepfilled', color='blue', label=f'TTJets > {str(bdt_cut)}')
            plt.hist(qcd_data[key][bdt_qcd_mask], bins=bins, weights=qcd_weight[bdt_qcd_mask], histtype='stepfilled', color='green', label=f'QCD > {str(bdt_cut)}')
        else:
            plt.hist(tt_data[key][bdt_tt_mask], bins=bins, weights=tt_weight[bdt_tt_mask], histtype='step', alpha=alpha, color='blue', label=f'TTJets > {str(bdt_cut)}')
            plt.hist(qcd_data[key][bdt_qcd_mask], bins=bins, weights=qcd_weight[bdt_qcd_mask], histtype='step', alpha=alpha, color='green', label=f'QCD > {str(bdt_cut)}')
            plt.hist(sig_data[key][bdt_sig_mask], bins=bins, weights=sig_weight[bdt_sig_mask], histtype='step', alpha=alpha, color='orange', label=f'Signal > {str(bdt_cut)}')
        plt.xlabel(f'{key}') #labelpad=30)
        plt.ylabel('Events Weighted for Training')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        nloops += 1
 
    logger.info(f'Saving to plots/input_features/{key}_BDT_cut_applied.png')
    # Save linear axis
    plt.savefig(f'plots/input_features/{key}_BDT_cut_applied.png', bbox_inches='tight')
    # Save log axis
    plt.gca().set_yscale('log')
    plt.savefig(f'plots/input_features/logscale_{key}_BDT_cut_applied.png', bbox_inches='tight')
    fig.clear()
