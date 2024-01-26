#==============================================================================
# plot_mZ_auc_comparions.py -------------------------------------------------------
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

    # QCD
    qcd_ROC_dict = { 
                 'mZ 200' : {'full window': 0.791516227180152,  'iterative': 0.7052874668978381, 'qcd=3tt=3e6 iterative': 0.6185459744645956, 'm(Zprime)<400': 0.7797526450845120}, 
                 'mZ 250' : {'full window': 0.7908284106475005, 'iterative': 0.6662696670693331, 'qcd=3tt=3e6 iterative': 0.6303099712823435, 'm(Zprime)<400': 0.7730533254143075},
                 'mZ 300' : {'full window': 0.7613581198560455, 'iterative': 0.6475435418391015, 'qcd=3tt=3e6 iterative': 0.6219583068577372, 'm(Zprime)<400': 0.7527227186937293},
                 'mZ 350' : {'full window': 0.7315129343369337, 'iterative': 0.6547643311776505, 'qcd=3tt=3e6 iterative': 0.6128860393747451, 'm(Zprime)<400': 0.7194738549268618},
                 'mZ 400' : {'full window': 0.700335766932438,  'iterative': 0.6782885235462466, 'qcd=3tt=3e6 iterative': 0.6081551693614586, 'm(Zprime)<400': 0.6975223420388295}, 
                 'mZ 450' : {'full window': 0.6850395338042998, 'iterative': 0.7033775731550349, 'qcd=3tt=3e6 iterative': 0.6059391926660088, 'm(Zprime)<400': 0.6829668400766487},
                 'mZ 500' : {'full window': 0.6503214167904827, 'iterative': 0.7182411473799015, 'qcd=3tt=3e6 iterative': 0.5688429486696006, 'm(Zprime)<400': 0.6595104699161872},
                 'mZ 550' : {'full window': 0.635718270799729,  'iterative': 0.7267146676720004, 'qcd=3tt=3e6 iterative': 0.5962606397930337, 'm(Zprime)<400': 0.6703795048370708}
                }

    # tt
    tt_ROC_dict = { 
                 'mZ 200' : {'full window': 0.657400767404975,  'iterative': 0.6728466253767816, 'qcd=3tt=3e6 iterative': 0.6119812451751700, 'm(Zprime)<400': 0.7766201681353790},
                 'mZ 250' : {'full window': 0.6755873769751104, 'iterative': 0.6469395234741973, 'qcd=3tt=3e6 iterative': 0.6372674964090255, 'm(Zprime)<400': 0.7831820128968622}, 
                 'mZ 300' : {'full window': 0.6692602123722782, 'iterative': 0.6367031405400014, 'qcd=3tt=3e6 iterative': 0.6506900376367746, 'm(Zprime)<400': 0.7911403205626494},
                 'mZ 350' : {'full window': 0.6318047774592488, 'iterative': 0.6217414685989808, 'qcd=3tt=3e6 iterative': 0.6436448577941927, 'm(Zprime)<400': 0.7680300531259030},  
                 'mZ 400' : {'full window': 0.5971220654254941, 'iterative': 0.6055996181120576, 'qcd=3tt=3e6 iterative': 0.6486942357669154, 'm(Zprime)<400': 0.7483648637256940}, 
                 'mZ 450' : {'full window': 0.5824760598908825, 'iterative': 0.6182807779503291, 'qcd=3tt=3e6 iterative': 0.6465608136466665, 'm(Zprime)<400': 0.7480671978685101}, 
                 'mZ 500' : {'full window': 0.5551714008529373, 'iterative': 0.6127370701225252, 'qcd=3tt=3e6 iterative': 0.6305892629589407, 'm(Zprime)<400': 0.7201539275248770},
                 'mZ 550' : {'full window': 0.5534466018063314, 'iterative': 0.6195535805458667, 'qcd=3tt=3e6 iterative': 0.6591825225919705, 'm(Zprime)<400': 0.7363614820975960}
                }

    qcd_ROC_plot_dict = {'full window': {}, 'iterative': {}, 'qcd=3tt=3e6 iterative': {}, 'm(Zprime)<400': {} }
    tt_ROC_plot_dict = {'full window': {}, 'iterative': {}, 'qcd=3tt=3e6 iterative': {}, 'm(Zprime)<400': {} }

    # need dictionary that has train_type, xval (eval_type), ROC val
    for eval_type, eval_dict in qcd_ROC_dict.items() :
        for train_type, ROC_val in eval_dict.items() : 
            qcd_ROC_plot_dict[train_type][eval_type] = ROC_val
    for eval_type, eval_dict in tt_ROC_dict.items() :
        for train_type, ROC_val in eval_dict.items() : 
            tt_ROC_plot_dict[train_type][eval_type] = ROC_val

    # Make one line on the chart for the training on each signal type
    ROC_dicts = [qcd_ROC_plot_dict, tt_ROC_plot_dict]
    nloops = 0
    for ROC_plot_dict in ROC_dicts :

        fig = plt.figure(figsize=(8,8))
        ax = fig.gca()
 
        # Options to make the plot fancier 
        hep.cms.label(rlabel="Work in Progress")
 
        # Y axis
        plt.ylabel("ROC AUC")
 
        # X axis and rotate values for readability
        plt.xlabel("Signals Evaluated On")
        plt.xticks(rotation=45, ha='right')

        for train_type, results in ROC_plot_dict.items() :
 
            # plot the BDT ROC values for a given training
            x = []
            y = []
            for eval_type, val in results.items() :
                x.append(eval_type)
                y.append(val)
            plt.plot(x,y, label=train_type, marker='o')
            
            #if (train_type == "all_signals") :
            #    plt.plot(x,y, label=train_type, marker='o', color='peru')
            #else :
            #    plt.plot(x,y, label=train_type, marker='o', alpha=0.2)
 
        # Put legend outside the plot
        legend_outside = plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', title="Model Type")

        if nloops == 0 : 
            outfile = "bdt_qcd_AUC_comparisons.png"
        else :
            outfile = "bdt_tt_AUC_comparisons.png"
        #osp.join(plotdir, f'bdt{bdtcut}_{"bkg" if is_bkg else "sig"}_mt_comparisons_{name}.png')
        #ax.set_title(f'bdt{bdtcut}_{name}')
        #ax.title(f'bdt{bdtcut}_{name}')
        logger.info(f'Saving to {outfile}')
        plt.savefig(outfile, bbox_inches='tight')
        #ax.clear()
        fig.clear()
        nloops +=1


if __name__ == '__main__':
    main()

