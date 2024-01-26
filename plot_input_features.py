#=========================================================================================
# plot_input_features.py -----------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Plot some input features for debugging purposes ----------------------------------------
#-----------------------------------------------------------------------------------------

import os, os.path as osp, glob, pickle, logging, argparse, sys, re, pprint
import numpy as np
import matplotlib
matplotlib.use('Agg') # prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS") # CMS plot style
from common import logger, DATADIR, Columns, time_and_log, columns_to_numpy, set_matplotlib_fontsizes, imgcat, add_key_value_to_json, filter_pt


# Define a list of .npz filenames
bkg_cols = [Columns.load(f) for f in glob.glob(DATADIR + '/train_bkg/Summer20UL18/TTJets_*.npz')]
sig_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*.npz')]

# Features to plot
features = [
    ['weight'], ['puweight'], ['girth'], ['ptd'], ['axismajor'], ['axisminor'],
    ['ecfm2b1'], ['ecfd2b1'], ['ecfc2b1'], ['ecfn2b2'], ['metdphi'],
    ['ak15_chad_ef'], ['ak15_nhad_ef'], ['ak15_elect_ef'], ['ak15_muon_ef'], ['ak15_photon_ef'], 
    ]

# Initialize an empty dictionary to store feature data
bkg_data = {}
sig_data = {}

# Load feature data from each .npz file and store it in the dictionary
weight = []
sig_weight = []
for feature in features :
    for cols in bkg_cols:
        this_bkg = cols.to_numpy(feature)
        feature_name = feature[0]

        if feature_name not in bkg_data:
            bkg_data[feature_name] = []
        if feature_name == 'weight' :
            #weight.append(this_bkg)
            len_bkg_cols = len(this_bkg)
            weight.append((1./len_bkg_cols)*np.ones(len_bkg_cols))
        bkg_data[feature_name].append(this_bkg)
    for cols in sig_cols:
        this_sig = cols.to_numpy(feature)
        feature_name = feature[0]

        if feature_name not in sig_data:
            sig_data[feature_name] = []
        if feature_name == 'weight' :
            len_sig_cols = len(this_sig)
            sig_weight.append((1./len_sig_cols)*np.ones(len_sig_cols))
        sig_data[feature_name].append(this_sig)

# Convert lists to numpy arrays
weight = np.concatenate(weight)
sig_weight = np.concatenate(sig_weight)
for key in bkg_data.keys():
    bkg_data[key] = np.concatenate(bkg_data[key])
for key in sig_data.keys():
    sig_data[key] = np.concatenate(sig_data[key])

# Set total sig weight equal to total bkg weight
sig_weight *= np.sum(weight) / np.sum(sig_weight)

# Plot features
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
for key in bkg_data.keys():

    # Calculate the bin edges from the data range
    n_bins = 50
    data_range = (min(np.min(bkg_data[key]), np.min(sig_data[key])), max(np.max(bkg_data[key]), np.max(sig_data[key])))
    bins = np.linspace(data_range[0], data_range[1], n_bins)

    # Plot the histograms
    hep.cms.label(rlabel="Work in Progress") # Options to make the plot fancier 
    plt.hist(bkg_data[key], bins=bins, weights=weight, histtype='step', color='blue', label='TTJets')
    plt.hist(sig_data[key], bins=bins, weights=sig_weight, histtype='step', color='orange', label='Signal')
    plt.xlabel(f'{key}')#, labelpad=30)
    plt.ylabel('Number of Events')
    plt.legend()

    logger.info(f'Saving to plots/input_features/{key}.png')
    plt.savefig(f'plots/input_features/{key}.png', bbox_inches='tight')
    fig.clear()
