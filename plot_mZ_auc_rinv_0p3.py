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
                 'mZ 200' : {'full window': 0.7981192190374868, 'iterative': 0.7274478992550253, 'qcd=3tt=3e6 iterative': 0.6257047582581562, 'iterative semi-visible forest': 0.7461959370944351, 'increasing iterative SVF': 0.6854101745267663 , 'non-iter SVF' : 0.7432124822618305 }, 
                 'mZ 250' : {'full window': 0.7918044534523965, 'iterative': 0.6818182031395331, 'qcd=3tt=3e6 iterative': 0.6383487426753415, 'iterative semi-visible forest': 0.7445841640216978, 'increasing iterative SVF': 0.6747220014208463 , 'non-iter SVF' : 0.7449195028139910 },
                 'mZ 300' : {'full window': 0.7726636462166224, 'iterative': 0.6596636657973965, 'qcd=3tt=3e6 iterative': 0.6425546765840626, 'iterative semi-visible forest': 0.7161196719681941, 'increasing iterative SVF': 0.6635609752037200 , 'non-iter SVF' : 0.7161196719681941 },
                 'mZ 350' : {'full window': 0.7413308780312082, 'iterative': 0.6606401315014750, 'qcd=3tt=3e6 iterative': 0.6389165377371847, 'iterative semi-visible forest': 0.6827809151071897, 'increasing iterative SVF': 0.6486284614810987 , 'non-iter SVF' : 0.7058861649477784 },
                 'mZ 400' : {'full window': 0.7253879775495686, 'iterative': 0.6833824983823712, 'qcd=3tt=3e6 iterative': 0.6354224441629877, 'iterative semi-visible forest': 0.6687523666362398, 'increasing iterative SVF': 0.6695591509716280 , 'non-iter SVF' : 0.6885129632171031 }, 
                 'mZ 450' : {'full window': 0.7063737424949023, 'iterative': 0.7136719915066633, 'qcd=3tt=3e6 iterative': 0.6293740762103313, 'iterative semi-visible forest': 0.6431550331601110, 'increasing iterative SVF': 0.6727841035796331 , 'non-iter SVF' : 0.6772002346436813 },
                 'mZ 500' : {'full window': 0.6735583567904657, 'iterative': 0.7349279070452404, 'qcd=3tt=3e6 iterative': 0.5983137282496429, 'iterative semi-visible forest': 0.6121494015763984, 'increasing iterative SVF': 0.6762221392021870 , 'non-iter SVF' : 0.6582315588035692 },
                 'mZ 550' : {'full window': 0.6608035438112277, 'iterative': 0.7341479223253055, 'qcd=3tt=3e6 iterative': 0.6188783282394406, 'iterative semi-visible forest': 0.6051714305720832, 'increasing iterative SVF': 0.6929926595045992 , 'non-iter SVF' : 0.6514565799096986 }
                }

    # tt
    tt_ROC_dict = { 
                 'mZ 200' : {'full window': 0.6620819016436601, 'iterative': 0.6947704336043975, 'qcd=3tt=3e6 iterative': 0.6214500934965583, 'iterative semi-visible forest': 0.7381495624117981, 'increasing iterative SVF': 0.7412713721557669, 'non-iter SVF' : 0.7132400571131481 },
                 'mZ 250' : {'full window': 0.6761436896901162, 'iterative': 0.6618018567670145, 'qcd=3tt=3e6 iterative': 0.6465353218720980, 'iterative semi-visible forest': 0.7446890619839527, 'increasing iterative SVF': 0.7565886139210641, 'non-iter SVF' : 0.7193036408472695 }, 
                 'mZ 300' : {'full window': 0.6794325586726541, 'iterative': 0.6487064470645150, 'qcd=3tt=3e6 iterative': 0.6725497425344281, 'iterative semi-visible forest': 0.7384252615293745, 'increasing iterative SVF': 0.7765770735739316, 'non-iter SVF' : 0.7194818150502191 },
                 'mZ 350' : {'full window': 0.6409057299258172, 'iterative': 0.6274431070434626, 'qcd=3tt=3e6 iterative': 0.6694721866764263, 'iterative semi-visible forest': 0.7131822334905510, 'increasing iterative SVF': 0.7729840056536132, 'non-iter SVF' : 0.7043541539420684 },  
                 'mZ 400' : {'full window': 0.6184884460851601, 'iterative': 0.6106816617264756, 'qcd=3tt=3e6 iterative': 0.6747735896380345, 'iterative semi-visible forest': 0.6968910899500670, 'increasing iterative SVF': 0.7880506781374976, 'non-iter SVF' : 0.6951919020629781 }, 
                 'mZ 450' : {'full window': 0.6021860077031883, 'iterative': 0.6300529642022792, 'qcd=3tt=3e6 iterative': 0.6696514667899169, 'iterative semi-visible forest': 0.6814090219795107, 'increasing iterative SVF': 0.7897440057740669, 'non-iter SVF' : 0.7004032545287222 }, 
                 'mZ 500' : {'full window': 0.5775492379291758, 'iterative': 0.6313464881982536, 'qcd=3tt=3e6 iterative': 0.6584424842036047, 'iterative semi-visible forest': 0.6554680597697209, 'increasing iterative SVF': 0.7861402500135270, 'non-iter SVF' : 0.7067210369990872 },
                 'mZ 550' : {'full window': 0.5759687232574333, 'iterative': 0.6263480846206931, 'qcd=3tt=3e6 iterative': 0.6800417256995792, 'iterative semi-visible forest': 0.6523898760317175, 'increasing iterative SVF': 0.7946949648277202, 'non-iter SVF' : 0.7085889217144454 }
                }

    qcd_ROC_plot_dict = {'full window': {}, 'iterative': {}, 'qcd=3tt=3e6 iterative': {}, 'iterative semi-visible forest': {}, 'increasing iterative SVF' : {}, 'non-iter SVF' : {} }
    tt_ROC_plot_dict  = {'full window': {}, 'iterative': {}, 'qcd=3tt=3e6 iterative': {}, 'iterative semi-visible forest': {}, 'increasing iterative SVF' : {}, 'non-iter SVF' : {} }

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

