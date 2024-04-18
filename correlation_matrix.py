#=========================================================================================
# correlation_matrix.py ------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Plot the correlation matrix of the training inputs for the BDT -------------------------
#-----------------------------------------------------------------------------------------

import os, os.path as osp, glob, pickle, logging, warnings, json, math, re
import numpy as np
import matplotlib
matplotlib.use('Agg') # prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
from common import DATADIR, Columns, columns_to_numpy_single
from matplotlib.colors import LinearSegmentedColormap
import mplhep as hep
hep.style.use("CMS") # CMS plot style

# Define the training features list
training_features = [
    'mt', 'girth', 'ptd', 'axismajor', 'axisminor',
    #'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef',
]

# grab correct signal files
signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_signal/*.npz')]

# grab background files
qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/QCD_*.npz')]
qcd_cols = list(filter(lambda c: c.metadata['ptbin'][0]>=300., qcd_cols))
ttjets_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/test_bkg/Summer20UL18/TTJets_*.npz')]
bkg_cols = qcd_cols + ttjets_cols
#bkg_cols = ttjets_cols

# Make arrays with the data
X, weight = columns_to_numpy_single(bkg_cols, features=training_features, mt_high = 650, mt_low = 180)

# Compute the correlation matrix
corrmat = np.corrcoef(X.T)

# Create a figure
fig = plt.figure(figsize=(8, 8))
ax = fig.gca()

# Define the custom colormap
cmap_colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Red, White, Blue
cmap = LinearSegmentedColormap.from_list('CustomCmap', cmap_colors, N=256)

# Display the correlation matrix using matshow
#mshow = ax.matshow(corrmat, cmap=plt.cm.Reds)
mshow = ax.matshow(corrmat, cmap=cmap, vmin=-1, vmax=1)

# Set labels for the axes
labels = training_features
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.xaxis.tick_bottom()  # Place x-axis ticks and ticklabels at bottom
ax.xaxis.set_label_position('bottom')  # Set x-axis labels position to bottom
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

# Add a title
#plt.title('Feature Correlation Matrix')
# Options to make the plot fancier 
hep.cms.label(rlabel="2018 (13 TeV)")

# Add colorbar to the right side
cbar_ax = fig.add_axes([0.93, 0.1, 0.05, 0.8])  # [x, y, width, height]
cbar = fig.colorbar(mshow, cax=cbar_ax)
cbar.ax.set_ylabel('Correlation')

# Plot feature importance
plt.savefig("plots/correlation_matrix.png", bbox_inches='tight', pad_inches=1.0)
plt.savefig("plots/correlation_matrix.pdf", bbox_inches='tight', pad_inches=1.0)


