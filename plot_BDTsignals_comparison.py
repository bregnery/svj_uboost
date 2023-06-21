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

    ROC_dict = { 'all_signals' : {'mdark1.0_rinv0.1': 0.7964630985591721, 'mdark1.0_rinv0.3': 0.7997457276656793, 'mdark1.0_rinv0.7': 0.7426469801678445, 'mdark5.0_rinv0.1': 0.8021262545783309, 'mdark5.0_rinv0.3': 0.8183792660024096, 'mdark5.0_rinv0.7': 0.8033954222069575, 'mdark10.0_rinv0.1': 0.8007889485615123, 'mdark10.0_rinv0.3': 0.8111999589067895, 'mdark10.0_rinv0.7': 0.8137496427249769, 'without_0.7': 0.8340610763358534, 'all_signals': 0.8340610763358534},
                 'without_0.7' : {'mdark1.0_rinv0.1': 0.8220775182438435, 'mdark1.0_rinv0.3': 0.8107906070156294, 'mdark1.0_rinv0.7': 0.7102129056055676, 'mdark5.0_rinv0.1': 0.8479255171040363, 'mdark5.0_rinv0.3': 0.8548408149157086, 'mdark5.0_rinv0.7': 0.7945881879285095, 'mdark10.0_rinv0.1': 0.8431797117996538, 'mdark10.0_rinv0.3': 0.8466362786079974, 'mdark10.0_rinv0.7': 0.8170830246805857, 'without_0.7': 0.8641194525181778, 'all_signals': 0.8641194525181778},
                 'mdark1_rinv0.1' : {'mdark1.0_rinv0.1': 0.8873549880380144, 'mdark1.0_rinv0.3': 0.8797783032179364, 'mdark1.0_rinv0.7': 0.7682659130250464, 'mdark5.0_rinv0.1': 0.8099534669506865, 'mdark5.0_rinv0.3': 0.8214192141500614, 'mdark5.0_rinv0.7': 0.7859776714337423, 'mdark10.0_rinv0.1': 0.7956617841109984, 'mdark10.0_rinv0.3': 0.8041798971910941, 'mdark10.0_rinv0.7': 0.7973414557676354, 'without_0.7' : 0.8622729254024689, 'all_signals': 0.8622729254024689},
                 'mdark1_rinv0.3' : {'mdark1.0_rinv0.1': 0.8685507534487733, 'mdark1.0_rinv0.3': 0.8810781040868596, 'mdark1.0_rinv0.7': 0.8285650146806376, 'mdark5.0_rinv0.1': 0.7789873067338585, 'mdark5.0_rinv0.3': 0.8021772386236575, 'mdark5.0_rinv0.7': 0.8239463650142818, 'mdark10.0_rinv0.1': 0.7637750134015703, 'mdark10.0_rinv0.3': 0.7804038194579029, 'mdark10.0_rinv0.7': 0.8119000575803149, 'without_0.7' : 0.8524762945357897, 'all_signals': 0.8524762945357897},
                 'mdark1_rinv0.7' : {'mdark1.0_rinv0.1': 0.7916430829075859, 'mdark1.0_rinv0.3': 0.8455128334505637, 'mdark1.0_rinv0.7': 0.9034148910908147, 'mdark5.0_rinv0.1': 0.6720608267763545, 'mdark5.0_rinv0.3': 0.7189880413236495, 'mdark5.0_rinv0.7': 0.8542049550000661, 'mdark10.0_rinv0.1': 0.66776284672422, 'mdark10.0_rinv0.3': 0.7030309406192226, 'mdark10.0_rinv0.7': 0.8130876353232996, 'without_0.7' : 0.7871352309683287, 'all_signals': 0.7871352309683287},
                 'mdark5_rinv0.1' : {'mdark1.0_rinv0.1': 0.7903950669265122, 'mdark1.0_rinv0.3': 0.7680878968894295, 'mdark1.0_rinv0.7': 0.6415630331764592, 'mdark5.0_rinv0.1': 0.8873994216370569, 'mdark5.0_rinv0.3': 0.8855140076621756, 'mdark5.0_rinv0.7': 0.7820786074851501, 'mdark10.0_rinv0.1': 0.8794272665730682, 'mdark10.0_rinv0.3': 0.8742755903045645, 'mdark10.0_rinv0.7': 0.8050191737189308, 'without_0.7' : 0.870164598606012, 'all_signals': 0.870164598606012},
                 'mdark5_rinv0.3' : {'mdark1.0_rinv0.1': 0.7838302125735245, 'mdark1.0_rinv0.3': 0.7744738216846339, 'mdark1.0_rinv0.7': 0.6801924058352264, 'mdark5.0_rinv0.1': 0.8567014177581527, 'mdark5.0_rinv0.3': 0.8626708173906351, 'mdark5.0_rinv0.7': 0.7988038989550837, 'mdark10.0_rinv0.1': 0.8493559719499083, 'mdark10.0_rinv0.3': 0.8492958050323166, 'mdark10.0_rinv0.7': 0.8117710095940073, 'without_0.7' : 0.8528451440891374, 'all_signals': 0.8528451440891374},
                 'mdark5_rinv0.7' : {'mdark1.0_rinv0.1': 0.7175610261188103, 'mdark1.0_rinv0.3': 0.7441241464147366, 'mdark1.0_rinv0.7': 0.7714565130776856, 'mdark5.0_rinv0.1': 0.7255410801979072, 'mdark5.0_rinv0.3': 0.7557441852725991, 'mdark5.0_rinv0.7': 0.8134398510386236, 'mdark10.0_rinv0.1': 0.7281924466070671, 'mdark10.0_rinv0.3': 0.7464643346416807, 'mdark10.0_rinv0.7': 0.7964929991021809, 'without_0.7' : 0.7636729032345355, 'all_signals': 0.7636729032345355}, 
                 'mdark10_rinv0.1' : {'mdark1.0_rinv0.1': 0.8096120027311485, 'mdark1.0_rinv0.3': 0.7836855892601237, 'mdark1.0_rinv0.7': 0.6608836872551085, 'mdark5.0_rinv0.1': 0.8968542671794594, 'mdark5.0_rinv0.3': 0.8954593186501897, 'mdark5.0_rinv0.7': 0.7893267123586826, 'mdark10.0_rinv0.1': 0.903846117008042, 'mdark10.0_rinv0.3': 0.901893536051634, 'mdark10.0_rinv0.7': 0.8430921705654169, 'without_0.7' : 0.8887070180813552, 'all_signals': 0.8887070180813552}, 
                 'mdark10_rinv0.3' : {'mdark1.0_rinv0.1': 0.7927220857452414, 'mdark1.0_rinv0.3': 0.7776399269549651, 'mdark1.0_rinv0.7': 0.6818073796611545, 'mdark5.0_rinv0.1': 0.857657222365058, 'mdark5.0_rinv0.3': 0.8618042930175923, 'mdark5.0_rinv0.7': 0.7873958723242648, 'mdark10.0_rinv0.1': 0.8670121177543816, 'mdark10.0_rinv0.3': 0.8697690236105229, 'mdark10.0_rinv0.7': 0.8333742808573209, 'without_0.7' : 0.858250734394396, 'all_signals': 0.858250734394396},
                 'mdark10_rinv0.7' : {'mdark1.0_rinv0.1': 0.7264986685431165, 'mdark1.0_rinv0.3': 0.7433309270320348, 'mdark1.0_rinv0.7': 0.7476739837086654, 'mdark5.0_rinv0.1': 0.733981281606635, 'mdark5.0_rinv0.3': 0.7616362779312952, 'mdark5.0_rinv0.7': 0.7953848662528744, 'mdark10.0_rinv0.1': 0.7520669729245338, 'mdark10.0_rinv0.3': 0.7714866832523235, 'mdark10.0_rinv0.7': 0.8116680020158212, 'without_0.7' : 0.7710248377108154, 'all_signals': 0.7710248377108154},
                }

    ROC_plot_dict = {'mdark1.0_rinv0.1': {}, 'mdark1.0_rinv0.3': {}, 'mdark1.0_rinv0.7': {}, 'mdark5.0_rinv0.1': {}, 'mdark5.0_rinv0.3': {}, 'mdark5.0_rinv0.7': {}, 'mdark10.0_rinv0.1': {}, 'mdark10.0_rinv0.3': {}, 'mdark10.0_rinv0.7': {}, 'without_0.7' : {}, 'all_signals': {} }

    # need dictionary that has train_type, xval (eval_type), ROC val
    for eval_type, eval_dict in ROC_dict.items() :
        for train_type, ROC_val in eval_dict.items() : 
            ROC_plot_dict[train_type][eval_type] = ROC_val

    print(ROC_plot_dict)
            
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

