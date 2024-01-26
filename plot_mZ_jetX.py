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
                 'mZ 200' : {'full window': 0.7981192190374868, 'iterative': 0.6257047582581562, 'semi-visible forest': 0.6854101745267663, 'm(Zprime)<400': 0.7797526450845120, '20x8 iterations' : 0.6521218318446135 }, 
                 'mZ 250' : {'full window': 0.7918044534523965, 'iterative': 0.6383487426753415, 'semi-visible forest': 0.6747220014208463, 'm(Zprime)<400': 0.7730533254143075, '20x8 iterations' : 0.6532001988617824 },
                 'mZ 300' : {'full window': 0.7726636462166224, 'iterative': 0.6425546765840626, 'semi-visible forest': 0.6635609752037200, 'm(Zprime)<400': 0.7527227186937293, '20x8 iterations' : 0.6482242465518853 },
                 'mZ 350' : {'full window': 0.7413308780312082, 'iterative': 0.6389165377371847, 'semi-visible forest': 0.6486284614810987, 'm(Zprime)<400': 0.7194738549268618, '20x8 iterations' : 0.6587469359240298 },
                 'mZ 400' : {'full window': 0.7253879775495686, 'iterative': 0.6354224441629877, 'semi-visible forest': 0.6695591509716280, 'm(Zprime)<400': 0.6975223420388295, '20x8 iterations' : 0.6625934839960581 }, 
                 'mZ 450' : {'full window': 0.7063737424949023, 'iterative': 0.6293740762103313, 'semi-visible forest': 0.6727841035796331, 'm(Zprime)<400': 0.6829668400766487, '20x8 iterations' : 0.6592005351797783 },
                 'mZ 500' : {'full window': 0.6735583567904657, 'iterative': 0.5983137282496429, 'semi-visible forest': 0.6762221392021870, 'm(Zprime)<400': 0.6595104699161872, '20x8 iterations' : 0.6604543200534827 },
                 'mZ 550' : {'full window': 0.6608035438112277, 'iterative': 0.6188783282394406, 'semi-visible forest': 0.6929926595045992, 'm(Zprime)<400': 0.6703795048370708, '20x8 iterations' : 0.6570041621475976 }
                }                                                                                                                                                              
                                                                                                                                                                               
    # tt                                                                                                                                                                       
    tt_ROC_dict = {                                                                                                                                                            
                 'mZ 200' : {'full window': 0.6620819016436601, 'iterative': 0.6214500934965583, 'semi-visible forest': 0.7412713721557669, 'm(Zprime)<400': 0.7766201681353790, '20x8 iterations' : 0.7042569434188284 },
                 'mZ 250' : {'full window': 0.6761436896901162, 'iterative': 0.6465353218720980, 'semi-visible forest': 0.7565886139210641, 'm(Zprime)<400': 0.7831820128968622, '20x8 iterations' : 0.7131824343408465 }, 
                 'mZ 300' : {'full window': 0.6794325586726541, 'iterative': 0.6725497425344281, 'semi-visible forest': 0.7765770735739316, 'm(Zprime)<400': 0.7911403205626494, '20x8 iterations' : 0.7273820422105332 },
                 'mZ 350' : {'full window': 0.6409057299258172, 'iterative': 0.6694721866764263, 'semi-visible forest': 0.7729840056536132, 'm(Zprime)<400': 0.7680300531259030, '20x8 iterations' : 0.7348976566842282 },  
                 'mZ 400' : {'full window': 0.6184884460851601, 'iterative': 0.6747735896380345, 'semi-visible forest': 0.7880506781374976, 'm(Zprime)<400': 0.7483648637256940, '20x8 iterations' : 0.7294079299789167 }, 
                 'mZ 450' : {'full window': 0.6021860077031883, 'iterative': 0.6696514667899169, 'semi-visible forest': 0.7897440057740669, 'm(Zprime)<400': 0.7480671978685101, '20x8 iterations' : 0.7294927519649368 }, 
                 'mZ 500' : {'full window': 0.5775492379291758, 'iterative': 0.6584424842036047, 'semi-visible forest': 0.7861402500135270, 'm(Zprime)<400': 0.7201539275248770, '20x8 iterations' : 0.7287345284749082 },
                 'mZ 550' : {'full window': 0.5759687232574333, 'iterative': 0.6800417256995792, 'semi-visible forest': 0.7946949648277202, 'm(Zprime)<400': 0.7363614820975960, '20x8 iterations' : 0.7149376357487084 }
                }

    qcd_ROC_plot_dict = {'full window': {}, 'iterative': {}, 'semi-visible forest' : {}, 'm(Zprime)<400': {}, '20x8 iterations': {}}
    tt_ROC_plot_dict  = {'full window': {}, 'iterative': {}, 'semi-visible forest' : {}, 'm(Zprime)<400': {}, '20x8 iterations': {}}

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

