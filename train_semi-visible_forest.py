import os, os.path as osp, glob, pickle, logging, argparse, sys, re, pprint
from time import strftime

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1001)

from common import logger, DATADIR, Columns, time_and_log, columns_to_numpy_for_training, columns_to_numpy_one_bkg, set_matplotlib_fontsizes, imgcat, add_key_value_to_json, filter_pt, mt_wind

# Training features
training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef', 
    ]
all_features = training_features + ['rho']

# Parameters for 'weak' BDT models 
# i.e. models on individual Z' mass points
# note: eta is learning rate
weak_params = dict( eta=0.05, min_child_weight=0.1, max_depth=6, subsample=1.0, n_estimators=400)
#weak_params = dict( eta=0.05, min_child_weight=0.1, max_depth=2, subsample=1.0, n_estimators=100)

# Parameters  for 'strong' BDT models
# i.e. models on the full mT window
strong_params = dict( eta=0.30, min_child_weight=0.1, max_depth=8, subsample=1.0, n_estimators=850)
#strong_params = dict( eta=0.05, min_child_weight=0.1, max_depth=2, subsample=1.0, n_estimators=100)

# Parameters for the ensembled BDT
ensem_params = dict( eta=0.05, min_child_weight=0.1, max_depth=6, subsample=1.0, n_estimators=400)
#ensem_params = dict( eta=0.05, min_child_weight=0.1, max_depth=2, subsample=1.0, n_estimators=100)


def print_weight_table(bkg_cols, signal_cols, weight_col='weight'):
    bkg_cols.sort(key=lambda s: (s.metadata['bkg_type'], s.metadata.get('ptbin',[0,0]), s.metadata.get('htbin',[0,0])))
    signal_cols.sort(key=lambda s: (s.metadata['mz'], s.metadata['rinv']))

    bkg_weights = []
    table = []

    for cols in bkg_cols:
        weight = cols.arrays[weight_col]
        weight_per_event = np.mean(weight) if len(weight) else -1.
        total_weight = np.sum(weight)
        bkg_weights.append(weight)
        table.append([
            osp.basename(cols.metadata['src']).replace('.npz',''),
            len(weight), weight_per_event, total_weight
            ])

    # Set signal weight equal to bkg weight

    total_bkg_weight = sum(np.sum(w) for w in bkg_weights)

    signal_weights = []
    for cols in signal_cols:
        signal_weights.append(np.ones(len(cols))/len(cols) if weight_col in {'manual', 'weight'} else cols.arrays[weight_col])

    total_signal_weight = sum(np.sum(w) for w in signal_weights)

    for weight, cols in zip(signal_weights, signal_cols):
        weight *= total_bkg_weight / total_signal_weight
        weight_per_event = np.mean(weight)
        total_weight = np.sum(weight)
        bkg_weights.append(weight)
        table.append([
            osp.basename(cols.metadata['src']).replace('.npz',''),
            len(weight), weight_per_event, total_weight
            ])

    total_weight = sum(r[-1] for r in table)    

    # Print the table
    width = max(len(r[0]) for r in table)
    print(f'{"Sample":{width}}')
    for r in table:
        print(f'{r[0]:{width}}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downsample', type=float, default=.4)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--node', type=str, help='Run training on a different lpc node.')
    parser.add_argument('--tag', type=str, help='Add some output to the output model file')
    # Some test flags for alternative trainings
    parser.add_argument('--gradientboost', action='store_true')
    parser.add_argument('--use_eta', action='store_true')
    # adding signal models
    parser.add_argument('--mdark', type=str)#, default='10.')
    parser.add_argument('--rinv', type=str)#, default='0.3')
    args, leftover_args = parser.parse_known_args()

    global training_features
    if args.use_eta:
        training_features.append('eta')

    if args.node:
        # Just get the first integer from the args.node string, and parse it to a valid lpc address
        print(args.node)
        node_nr = re.search(r'\d+', args.node).group()
        # Delete the --node argument from the command line
        args = sys.argv[:]
        i = args.index('--node'); args.pop(i+1); args.pop(i)
        # Submit the exact same command on a different node
        cmd = f'ssh -o StrictHostKeyChecking=no cmslpc{node_nr}.fnal.gov "source /uscms/home/bregnery/nobackup/miniconda3/etc/profile.d/conda.sh; cd /uscms/home/bregnery/nobackup/svj_uboost; conda activate bdtenv; nohup python ' + ' '.join(args) + '"'
        logger.info(f'Executing: {cmd}')
        os.system(cmd)
        return

    # Add a file logger for easier monitoring
    if not args.dry:
        if not osp.isdir('logs'): os.makedirs('logs')
        fmt = f'[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s {os.uname()[1]}] %(message)s'
        file_handler = logging.FileHandler(strftime('logs/log_train_%b%d.txt'))
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    logger.info(f'Running training script; args={args}')

    qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_bkg/Summer20UL18/QCD_*.npz')]
    tt_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_bkg/Summer20UL18/TTJets_*.npz')]
    bkg_cols = qcd_cols + tt_cols

    # Throw away the very low QCD bins (very low number of events)
    logger.info('Using QCD bins starting from pt>=300')
    # bkg_cols = list(filter(lambda cols: cols.metadata['bkg_type']!='qcd' or cols.metadata['ptbin'][0]>=300., bkg_cols))
    qcd_cols = filter_pt(qcd_cols, 300.)
    tt_cols = filter_pt(tt_cols, 300.)
    bkg_cols = filter_pt(bkg_cols, 300.)

    logger.info(f'Training features: {training_features}')

    # Parse the leftover arguments to see if there are any hyper parameters
    hyperpar_parser = argparse.ArgumentParser()
    hyperpar_args = hyperpar_parser.parse_args(leftover_args)

    import xgboost as xgb
    qcd_models = {} # signal vs. qcd models
    tt_models  = {} # signal vs. tt models

    # Perform an iterative training, reweighting not yet available
    # fit once per signal mass window on limited signal masses,
    # then perform a training on the full window
    outfile = strftime('models/svjbdt_%b%d_semi-visible_forest.json')
    mz_prime = [200, 250, 300, 350, 400, 450, 500, 550]

    if args.use_eta: outfile = outfile.replace('.json', '_eta.json')
    if args.tag: outfile = outfile.replace('.json', f'_{args.tag}.json')

    if args.dry:
        logger.info('Dry mode: Quitting')
        return

    # loop over the signal Z' masses
    for mz in mz_prime :

        # create a weak model for every Z' mass point for qcd and tt
        qcd_model = xgb.XGBClassifier(use_label_encoder=False, **weak_params)
        tt_model = xgb.XGBClassifier(use_label_encoder=False, **weak_params)

        # define mass window
        mt_window = [mz - 100, mz + 100] 
        
        # grab corresponding signal files
        signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*mz' + str(mz) + '*.npz')]
 
        # dark options need to be updated
        if args.mdark:
            signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*mdark'+args.mdark+'*.npz')]
        if args.rinv:
            signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*rinv'+args.rinv+'*.npz')]
        print_weight_table(bkg_cols, signal_cols, 'weight')

        # Start with signal vs QCD 
        # Apply mass window
        X, y, weight = columns_to_numpy_one_bkg(
            signal_cols, qcd_cols, training_features,
            downsample=args.downsample,
            mt_high = mt_window[1], mt_low = mt_window[0]
            )
 
        logger.info(f'Using {len(y)} events ({np.sum(y==1)} signal events, {np.sum(y==0)} bkg events)')
 
        # fit once per signal mass window on limited signal masses,
        with time_and_log(f'Begin training, signal vs qcd for mZ={mz}. This can take a while...'):
            qcd_model.fit(X, y, sample_weight=weight)

        # Add model to the dictionary
        qcd_models.update({"sig_qcd_mZ"+str(mz) : qcd_model})

        # Now train with signal vs tt
        # Apply mass window
        X, y, weight = columns_to_numpy_one_bkg(
            signal_cols, tt_cols, training_features,
            downsample=args.downsample,
            mt_high = mt_window[1], mt_low = mt_window[0]
            )
 
        logger.info(f'Using {len(y)} events ({np.sum(y==1)} signal events, {np.sum(y==0)} bkg events)')
 
        # fit once per signal mass window on limited signal masses,
        with time_and_log(f'Begin training, signal vs tt for mZ={mz}. This can take a while...'):
            tt_model.fit(X, y, sample_weight=weight)

        # Add model to the dictionary
        tt_models.update({"sig_tt_mZ"+str(mz) : tt_model})


    # grab all signal files
    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR+'/train_signal/*.npz')]

    # Start with the QCD model on the full window
    # make the model with 'strong' parameters
    full_qcd_model = xgb.XGBClassifier(use_label_encoder=False, **strong_params)

    # Apply full mass window (180 to 650)
    X, y, weight = columns_to_numpy_one_bkg(
        signal_cols, qcd_cols, training_features,
        downsample=args.downsample,
        )

    # fit over the full window (180 to 650)
    with time_and_log(f'Begin training, signal vs qcd model on 180 to 650 window. This can take a while...'):
        full_qcd_model.fit(X, y, sample_weight=weight)
   
    # Add model to the dictionary 
    qcd_models.update({"sig_qcd_180_to_650" : full_qcd_model})

    # Now train signal vs. tt model on the full window
    # make the model with 'strong' parameters
    full_tt_model = xgb.XGBClassifier(use_label_encoder=False, **strong_params)

    # Apply full mass window (180 to 650)
    X, y, weight = columns_to_numpy_one_bkg(
        signal_cols, tt_cols, training_features,
        downsample=args.downsample,
        )

    # fit over the full window (180 to 650)
    with time_and_log(f'Begin training, signal vs tt model on 180 to 650 window. This can take a while...'):
        full_tt_model.fit(X, y, sample_weight=weight)
   
    # Add model to the dictionary 
    tt_models.update({"sig_tt_180_to_650" : full_tt_model})

    # Now ensemble everything together

    # Apply full mass window (180 to 650)
    X, y, weight = columns_to_numpy_for_training(
        signal_cols, qcd_cols, tt_cols, training_features,
        downsample=args.downsample,
        )

    # Evaluate all models
    predictions = []
    prediction_names = []
    models = {**qcd_models, **tt_models} # Models will always need to be evaluated in this order
    for name, model in models.items():
        predictions.append(model.predict_proba(X)[:, 1] )
        prediction_names.append(name)

    # Combine all preditions into training features
    ensemble_features = np.column_stack(predictions)

    # The ensemble model settings
    ensemble_model = xgb.XGBClassifier(use_label_encoder=False, **ensem_params)

    # Train ensembled model
    # over the full window (180 to 650)
    with time_and_log(f'Begin training ensembled model. This can take a while...'):
        ensemble_model.fit(ensemble_features, y, sample_weight=weight)

    # save json output files of each model
    if not osp.isdir('models'): os.makedirs('models')
    if not osp.isdir('models/ensemble'): os.makedirs('models/ensemble')
    for name, model in models.items():
        outfile = strftime('models/ensemble/bdt_%b%d'+name+'.json')
        model.save_model(outfile)
        logger.info(f'Dumped trained model: '+name)
        add_key_value_to_json(outfile, 'features', training_features)

    outfile = strftime('models/ensemble/ensebled_%b%d_semi-visible_forest.json')
    model.save_model(outfile)
    logger.info(f'Dumped trained models')
    add_key_value_to_json(outfile, 'features', prediction_names)


if __name__ == '__main__':
    main()
