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
                 'mZ 200' : {'iterative Semi-visible Forest': 0.6800282926722331, 'original Semi-visible Forest': 0.732348704180949, 'TT=QCD=1e6 full window': 0.7569092658176908, 'TT=QCD=1e6 iterative': 0.6736019606705879, 'normal full window': 0.791516227180152, 'iterative w/ normal weights': 0.7052874668978381}, 
                 'mZ 250' : {'iterative Semi-visible Forest': 0.6674256925353568, 'original Semi-visible Forest': 0.7387162578502525, 'TT=QCD=1e6 full window': 0.7577934210490984, 'TT=QCD=1e6 iterative': 0.6458861656042498, 'normal full window': 0.7908284106475005, 'iterative w/ normal weights': 0.6662696670693331},
                 'mZ 300' : {'iterative Semi-visible Forest': 0.6422983591381883, 'original Semi-visible Forest': 0.7180936464486569, 'TT=QCD=1e6 full window': 0.7222923642318628, 'TT=QCD=1e6 iterative': 0.604740235323638, 'normal full window': 0.7613581198560455, 'iterative w/ normal weights': 0.6475435418391015},
                 'mZ 350' : {'iterative Semi-visible Forest': 0.6313146610179717, 'original Semi-visible Forest': 0.6927654737714479, 'TT=QCD=1e6 full window': 0.6924304392461512, 'TT=QCD=1e6 iterative': 0.5836807920714804, 'normal full window': 0.7315129343369337, 'iterative w/ normal weights': 0.6547643311776505},
                 'mZ 400' : {'iterative Semi-visible Forest': 0.6274709824395841, 'original Semi-visible Forest': 0.65550736779327, 'TT=QCD=1e6 full window': 0.656161650322619, 'TT=QCD=1e6 iterative': 0.5791817857449704, 'normal full window': 0.700335766932438, 'iterative w/ normal weights': 0.6782885235462466}, 
                 'mZ 450' : {'iterative Semi-visible Forest': 0.6411706055758142, 'original Semi-visible Forest': 0.65705679884111, 'TT=QCD=1e6 full window': 0.6462492683762007, 'TT=QCD=1e6 iterative': 0.5922513161149191, 'normal full window': 0.6850395338042998, 'iterative w/ normal weights': 0.7033775731550349},
                 'mZ 500' : {'iterative Semi-visible Forest': 0.640517346788467, 'original Semi-visible Forest': 0.6228092035334627, 'TT=QCD=1e6 full window': 0.6214181624986131, 'TT=QCD=1e6 iterative': 0.5910084602113603, 'normal full window': 0.6503214167904827, 'iterative w/ normal weights': 0.7182411473799015},
                 'mZ 550' : {'iterative Semi-visible Forest': 0.6510506763815594, 'original Semi-visible Forest': 0.6224765048007038, 'TT=QCD=1e6 full window': 0.6255728265811636, 'TT=QCD=1e6 iterative': 0.6165788847944947, 'normal full window': 0.635718270799729, 'iterative w/ normal weights': 0.7267146676720004}
                }

    # tt
    tt_ROC_dict = { 
                 'mZ 200' : {'iterative Semi-visible Forest': 0.7342373639656427, 'original Semi-visible Forest': 0.6978887930052052, 'TT=QCD=1e6 full window': 0.7560393459522684, 'TT=QCD=1e6 iterative': 0.6685499406291151, 'normal full window': 0.6313659211335089, 'iterative w/ normal weights': 0.6462033954025462},
                 'mZ 250' : {'iterative Semi-visible Forest': 0.7472012580125229, 'original Semi-visible Forest': 0.7099214899537936, 'TT=QCD=1e6 full window': 0.7791230069575722, 'TT=QCD=1e6 iterative': 0.6911711388511431, 'normal full window': 0.6495102884703341, 'iterative w/ normal weights': 0.6290184950008337}, 
                 'mZ 300' : {'iterative Semi-visible Forest': 0.7523444674660907, 'original Semi-visible Forest': 0.7084207207591421, 'TT=QCD=1e6 full window': 0.7828259816147194, 'TT=QCD=1e6 iterative': 0.6995266177974773, 'normal full window': 0.6435960181601702, 'iterative w/ normal weights': 0.6220447691764747},
                 'mZ 350' : {'iterative Semi-visible Forest': 0.7593345302051829, 'original Semi-visible Forest': 0.7039828692722152, 'TT=QCD=1e6 full window': 0.7911921536085842, 'TT=QCD=1e6 iterative': 0.7044850530879253, 'normal full window': 0.631059692536145, 'iterative w/ normal weights': 0.6136953108344346},  
                 'mZ 400' : {'iterative Semi-visible Forest': 0.7626633070861899, 'original Semi-visible Forest': 0.6898811792826874, 'TT=QCD=1e6 full window': 0.7906146376138368, 'TT=QCD=1e6 iterative': 0.7097845891524902, 'normal full window': 0.6136756221205769, 'iterative w/ normal weights': 0.6057330948995218}, 
                 'mZ 450' : {'iterative Semi-visible Forest': 0.7764680308447419, 'original Semi-visible Forest': 0.7067351279919394, 'TT=QCD=1e6 full window': 0.801464015132678, 'TT=QCD=1e6 iterative': 0.720495291526275, 'normal full window': 0.6095329803388757, 'iterative w/ normal weights': 0.6065752780199936}, 
                 'mZ 500' : {'iterative Semi-visible Forest': 0.7740751454530397, 'original Semi-visible Forest': 0.6911535561565337, 'TT=QCD=1e6 full window': 0.7939892224587974, 'TT=QCD=1e6 iterative': 0.7134207879468947, 'normal full window': 0.5828198812928307, 'iterative w/ normal weights': 0.6053907102034322},
                 'mZ 550' : {'iterative Semi-visible Forest': 0.7765450612249775, 'original Semi-visible Forest': 0.6974734728170622, 'TT=QCD=1e6 full window': 0.7946422101052583, 'TT=QCD=1e6 iterative': 0.7204547623266855, 'normal full window': 0.5683502337603777, 'iterative w/ normal weights': 0.6105955726296157}
                }

    qcd_ROC_plot_dict = {'iterative Semi-visible Forest': {},'original Semi-visible Forest': {}, 'TT=QCD=1e6 full window': {}, 'TT=QCD=1e6 iterative': {}, 'normal full window': {}, 'iterative w/ normal weights': {} }
    tt_ROC_plot_dict = {'iterative Semi-visible Forest': {},'original Semi-visible Forest': {}, 'TT=QCD=1e6 full window': {}, 'TT=QCD=1e6 iterative': {}, 'normal full window': {}, 'iterative w/ normal weights': {} }

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

