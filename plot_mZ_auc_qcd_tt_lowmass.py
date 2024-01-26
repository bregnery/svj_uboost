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
                 'mZ 200' : {'incr iter SVF': 0.6854101745267667, 'QCD < 400': 0.7865194146843691, 'tt < 400': 0.6346761982179021, 'many iter': 0.6521218318446133, 'QCD': 0.6657170970209152, 'tt': 0.6688404685582978}, 
                 'mZ 250' : {'incr iter SVF': 0.6747220014208465, 'QCD < 400': 0.7968416676511921, 'tt < 400': 0.662536060074139, 'many iter': 0.65320019886178240, 'QCD': 0.6682967325835146, 'tt': 0.6551798034598317},
                 'mZ 300' : {'incr iter SVF': 0.6635609752037201, 'QCD < 400': 0.7823565039535707, 'tt < 400': 0.6692191458941472, 'many iter': 0.6482242465518853, 'QCD': 0.6845512771927903, 'tt': 0.6510401318513882},
                 'mZ 350' : {'incr iter SVF': 0.6486284614810987, 'QCD < 400': 0.7474851002363558, 'tt < 400': 0.6625221194449460, 'many iter': 0.6587469359240298, 'QCD': 0.7035028835654840, 'tt': 0.6436514859995176},
                 'mZ 400' : {'incr iter SVF': 0.6695591509716280, 'QCD < 400': 0.7154447931778039, 'tt < 400': 0.6488122028926627, 'many iter': 0.6625934839960581, 'QCD': 0.7320695260785645, 'tt': 0.6323398222467438}, 
                 'mZ 450' : {'incr iter SVF': 0.6727841035796331, 'QCD < 400': 0.6876662702921099, 'tt < 400': 0.6473813412660963, 'many iter': 0.6592005351797783, 'QCD': 0.7286400660893516, 'tt': 0.6399238058994687},
                 'mZ 500' : {'incr iter SVF': 0.6762221392021870, 'QCD < 400': 0.6424007560311908, 'tt < 400': 0.65195201757008250, 'many iter': 0.6604543200534827, 'QCD': 0.7312531461940126, 'tt': 0.6353688892358110},
                 'mZ 550' : {'incr iter SVF': 0.6929926595045992, 'QCD < 400': 0.6265962839716874, 'tt < 400': 0.6657446864125182, 'many iter': 0.6570041621475976, 'QCD': 0.7094099343255563, 'tt': 0.6510467255154545}
                }

    # tt
    tt_ROC_dict = { 
                 'mZ 200' : {'incr iter SVF': 0.7412713721557669, 'QCD < 400': 0.6793997900932005, 'tt < 400': 0.8015691352368602, 'many iter': 0.7042569434188284, 'QCD': 0.6829243052073068, 'tt': 0.8140254718566967}, 
                 'mZ 250' : {'incr iter SVF': 0.7565886139210641, 'QCD < 400': 0.6953966874247690, 'tt < 400': 0.8229079575458236, 'many iter': 0.7131824343408465, 'QCD': 0.6773342313407500, 'tt': 0.8075007439775835},
                 'mZ 300' : {'incr iter SVF': 0.7765770735739316, 'QCD < 400': 0.6943913490735499, 'tt < 400': 0.8380598969077125, 'many iter': 0.7273820422105332, 'QCD': 0.6816144684190425, 'tt': 0.8015288206006862},
                 'mZ 350' : {'incr iter SVF': 0.7729840056536132, 'QCD < 400': 0.6467192754743684, 'tt < 400': 0.8318692862896893, 'many iter': 0.7348976566842282, 'QCD': 0.6676792763990783, 'tt': 0.7750305791616819},
                 'mZ 400' : {'incr iter SVF': 0.7880506781374976, 'QCD < 400': 0.5980374053357813, 'tt < 400': 0.8226349303244848, 'many iter': 0.7294079299789167, 'QCD': 0.6606164927924592, 'tt': 0.7449806042394020},
                 'mZ 450' : {'incr iter SVF': 0.7897440057740669, 'QCD < 400': 0.5660937584637415, 'tt < 400': 0.8223114802182803, 'many iter': 0.7294927519649368, 'QCD': 0.6495306705130446, 'tt': 0.7432702208341092},
                 'mZ 500' : {'incr iter SVF': 0.7861402500135270, 'QCD < 400': 0.5280718958117745, 'tt < 400': 0.8084968474035934, 'many iter': 0.7287345284749082, 'QCD': 0.6433967626877398, 'tt': 0.7213008980829743},
                 'mZ 550' : {'incr iter SVF': 0.7946949648277202, 'QCD < 400': 0.5289966746635875, 'tt < 400': 0.8140568326063016, 'many iter': 0.7149376357487084, 'QCD': 0.6303141647614656, 'tt': 0.7371221583169805}
                }

    qcd_ROC_plot_dict = {'incr iter SVF': {}, 'QCD < 400': {}, 'tt < 400': {}, 'many iter': {}, 'QCD': {}, 'tt': {} }
    tt_ROC_plot_dict  = {'incr iter SVF': {}, 'QCD < 400': {}, 'tt < 400': {}, 'many iter': {}, 'QCD': {}, 'tt': {} }

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

