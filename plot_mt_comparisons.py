#==============================================================================
# plot_mt_comparions.py -------------------------------------------------------
#------------------------------------------------------------------------------
# The goal of this plotting script is to be able to compare different bdt -----
#   trainings by looking at the mt distribution -------------------------------
#------------------------------------------------------------------------------

import os, os.path as osp, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import mplhep as hep
hep.style.use("CMS") # CMS plot style
from common import logger

#------------------------------------------------------------------------------
# User defined Functions ------------------------------------------------------
#------------------------------------------------------------------------------

# From matplotlib website https://matplotlib.org/stable/gallery/statistics/errorbars_and_boxes.html
def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='silver',
                     edgecolor='none', alpha=0.5):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe, y - ye), xe*2, ye*2)
                  for x, y, xe, ye in zip(xdata, ydata, xerror, yerror)]
                  #for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror,
                          fmt='none', ecolor='silver')

    return artists

#------------------------------------------------------------------------------
# Hardcoded inputs/options ----------------------------------------------------
#------------------------------------------------------------------------------

# define the histogram files of each model for comparison as a global variable
models = {
    'unreweighted'   : 'histograms_svjbdt_May11_allfiles_noreweight.json',
    'mt_reweighted'  : 'histograms_svjbdt_May10_allrinv_mTreweight_refmz250rinv0p3.json',
    'rho_reweighted' : 'histograms_svjbdt_May10_rinv0p3_rhoreweight_refmz250rinv0p3.json'
    }

#------------------------------------------------------------------------------
# User defined Functions ------------------------------------------------------
#------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonfile', type=str)
    args = parser.parse_args()

    plotdir = 'plots_' + args.jsonfile.replace('.json','')
    if not osp.isdir(plotdir): os.makedirs(plotdir)

    with open(args.jsonfile, 'r') as f:
        d = json.load(f)

    d = d['histograms']

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    # bdtcut is the working point
    # histograms is the data structure containing the mt distributions
    # metadata contains important information about the structure
    for bdtcut, histograms in d.items():
        #save background histogram for each working point
        bkg_hist = histograms['bkg']

        # mT comparisons
        # name is the sample name
        for name, hist in histograms.items():
            is_bkg = 'mz' not in hist['metadata']

            # Split into two plots and grab the first frame
            frame1 = fig.add_axes((.15,.3,.8,.6))

            # ax step means
            frame1.step(bkg_hist['binning'][:-1], bkg_hist['vals'], where='pre', label='bkg')
            frame1.step(hist['binning'][:-1], hist['vals'], where='mid', label=f'{name}')


            # Options to make the plot fancier 
            hep.cms.label(rlabel="Work in Progress")

            # log scale
            plt.yscale('log')
            plt.ylabel("Events")

            # Put legend outside the plot
            legend_outside = plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left')

            # store bin centers and for use in making the data points
            bin_centers = (np.array(bkg_hist['binning'][:-1]) + np.array(bkg_hist['binning'][1:]) ) / 2
            bin_width = bin_centers - np.array(bkg_hist['binning'][:-1])

            # get the x-axis limits to use in the residuals
            lowLim, upLim = plt.xlim() 

            # Make bottom plot for residuals (or s/sqrt(B) in this case)
            frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
            frame2=fig.add_axes((.15,.1,.8,.2))     
 
            # Make s/sqrt(B) plot
            _ = make_error_boxes(frame2, bin_centers, np.ones(len(bin_centers)), bin_width, np.zeros(len(hist['vals'])) )
            ssqrtb = hist['vals'] / np.sqrt(bkg_hist['vals']) 
            ssqrtbErr = np.sqrt(hist['vals'])/np.sqrt(np.sqrt(bkg_hist['vals']))
            plt.errorbar(bin_centers, ssqrtb, yerr=ssqrtbErr, fmt='ok')

            # Residual y axis
            plt.ylabel("S/$\sqrt(B)")

            # set the proper axis label on the left
            plt.xlabel("$m_{T}$ [GeV]", horizontalalignment='right', x=1.0)

            # match x-axis to histogram
            plt.xlim(lowLim, upLim)

            outfile = osp.join(plotdir, f'bdt{bdtcut}_{"bkg" if is_bkg else "sig"}_mt_comparisons_{name}.png')
            #ax.set_title(f'bdt{bdtcut}_{name}')
            #ax.title(f'bdt{bdtcut}_{name}')
            logger.info(f'Saving to {outfile}')
            plt.savefig(outfile, bbox_inches='tight')
            #ax.clear()
            fig.clear()


if __name__ == '__main__':
    main()
