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

def main():

#    plotdir = 'plots_' + args.jsonfile.replace('.json','')
#    if not osp.isdir(plotdir): os.makedirs(plotdir)

    ROC_dict = { 'all_signals' : {'mdark10.0_rinv0.1': 0.7987123739252041, 'mdark10.0_rinv0.3': 0.8066757090350122, 'mdark1.0_rinv0.1': 0.8057395851458233, 'mdark1.0_rinv0.3': 0.8035567859966879, 'mdark5.0_rinv0.1': 0.8061589507641997, 'mdark5.0_rinv0.3': 0.8183604094852863, 'all_signals': 0.8313766449929403},
                 'mdark1_rinv0.1' : {'mdark1.0_rinv0.1': 0.8585895850344737, 'mdark1.0_rinv0.3': 0.8470784487483162, 'mdark5.0_rinv0.1': 0.8164390833534751, 'mdark5.0_rinv0.3': 0.8258840445486427, 'mdark10.0_rinv0.1': 0.7888473871840734, 'mdark10.0_rinv0.3': 0.7935695499931597, 'all_signals': 0.8390624752363441},
                 'mdark1_rinv0.3' : {'mdark1.0_rinv0.1': 0.8420224088853437, 'mdark1.0_rinv0.3': 0.8580345591490418, 'mdark5.0_rinv0.1': 0.7856428348819274, 'mdark5.0_rinv0.3': 0.804765862851093, 'mdark10.0_rinv0.1': 0.7576228935219199, 'mdark10.0_rinv0.3': 0.7690744720236772, 'all_signals': 0.8358978262141714},
                 'mdark1_rinv0.7' : {'mdark1.0_rinv0.1': 0.7640668308818777, 'mdark1.0_rinv0.3': 0.8277157270599647, 'mdark5.0_rinv0.1': 0.683906654629703, 'mdark5.0_rinv0.3': 0.7207260250473931, 'mdark10.0_rinv0.1': 0.6662385002154871, 'mdark10.0_rinv0.3': 0.6923723128547737, 'all_signals': 0.8023045127061177},
                 'mdark5_rinv0.1' : {'mdark1.0_rinv0.1': 0.8409019854635424, 'mdark1.0_rinv0.3': 0.8027626796776719, 'mdark5.0_rinv0.1': 0.8801264112740724, 'mdark5.0_rinv0.3': 0.8785600478083709, 'mdark10.0_rinv0.1': 0.8757357081938978, 'mdark10.0_rinv0.3': 0.8733338414556595, 'all_signals': 0.8630205634835887},
                 'mdark5_rinv0.3' : {'mdark1.0_rinv0.1': 0.8218400960565251, 'mdark1.0_rinv0.3': 0.8024753676145777, 'mdark5.0_rinv0.1': 0.8501926853867213, 'mdark5.0_rinv0.3': 0.854240957665058, 'mdark10.0_rinv0.1': 0.846069926516653, 'mdark10.0_rinv0.3': 0.8473277234309722, 'all_signals': 0.845816933627412},
                 'mdark5_rinv0.7' : {'mdark1.0_rinv0.1': 0.7402958755043206, 'mdark1.0_rinv0.3': 0.7663497952920088, 'mdark5.0_rinv0.1': 0.7294551533658574, 'mdark5.0_rinv0.3': 0.7525700045783843, 'mdark10.0_rinv0.1': 0.7297277655335732, 'mdark10.0_rinv0.3': 0.7449776492330762, 'all_signals': 0.7783397228015773},
                 'mdark10_rinv0.1' : {'mdark1.0_rinv0.1': 0.8380703019238481, 'mdark1.0_rinv0.3': 0.7959816707972072, 'mdark5.0_rinv0.1': 0.8993658916622014, 'mdark5.0_rinv0.3': 0.8977821560489213, 'mdark10.0_rinv0.1': 0.902473121605852, 'mdark10.0_rinv0.3': 0.9010562345045625, 'all_signals': 0.8835898777480969},
                 'mdark10_rinv0.3' : {'mdark1.0_rinv0.1': 0.8123175458707287, 'mdark1.0_rinv0.3': 0.7855375671431768, 'mdark5.0_rinv0.1': 0.8634430678275415, 'mdark5.0_rinv0.3': 0.8653862888726003, 'mdark10.0_rinv0.1': 0.8666988714376401, 'mdark10.0_rinv0.3': 0.8686278118046873, 'all_signals': 0.8544729008027351},
                 'mdark10_rinv0.7' : {'mdark1.0_rinv0.1': 0.7335516366919208, 'mdark1.0_rinv0.3': 0.7460752584883947, 'mdark5.0_rinv0.1': 0.7468587744964783, 'mdark5.0_rinv0.3': 0.7653282979472781, 'mdark10.0_rinv0.1': 0.7549971911179114, 'mdark10.0_rinv0.3': 0.7697417860146979, 'all_signals': 0.779884992316567},
                }

    ROC_plot_dict = {'mdark1.0_rinv0.1': {}, 'mdark1.0_rinv0.3': {}, 'mdark5.0_rinv0.1': {}, 'mdark5.0_rinv0.3': {}, 'mdark10.0_rinv0.1': {}, 'mdark10.0_rinv0.3': {}, 'all_signals': {} }

    # need dictionary that has train_type, xval (eval_type), ROC val
    for eval_type, eval_dict in ROC_dict.items() :
        for train_type, ROC_val in eval_dict.items() : 
            ROC_plot_dict[train_type][eval_type] = ROC_val
            
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    # Options to make the plot fancier 
    hep.cms.label(rlabel="Work in Progress")

    # Y axis
    plt.ylabel("ROC AUC")

    # X axis and rotate values for readability
    plt.xlabel("Signals Evaluated On")
    plt.xticks(rotation=45, ha='right')

    # Make one line on the chart for the training on each signal type
    for train_type, results in ROC_plot_dict.items() :

        # plot the BDT ROC values for a given training
        x = []
        y = []
        for eval_type, val in results.items() :
            x.append(eval_type)
            y.append(val)
        plt.plot(x,y, label=train_type, marker='o')

    # Put legend outside the plot
    legend_outside = plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', title="Signals Trained On")

    outfile = "bdt_signaltrain_comparisons.png"
    #osp.join(plotdir, f'bdt{bdtcut}_{"bkg" if is_bkg else "sig"}_mt_comparisons_{name}.png')
    #ax.set_title(f'bdt{bdtcut}_{name}')
    #ax.title(f'bdt{bdtcut}_{name}')
    logger.info(f'Saving to {outfile}')
    plt.savefig(outfile, bbox_inches='tight')
    #ax.clear()
    fig.clear()


if __name__ == '__main__':
    main()

