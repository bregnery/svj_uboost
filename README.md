# About this repository

This is a repository that can train a uBoost or xgboost BDT for the SVJ boosted analysis.
It uses as much training data as it can, by using the precalculated TreeMaker Weight column.


## Setup

```
conda create -n bdtenv python=3.10
conda activate bdtenv  # Needed every time

conda install xgboost

pip install pandas
pip install requests
pip install numpy
pip install matplotlib
pip install tqdm
pip install numba

pip install git+ssh://git@github.com/boostedsvj/svj_ntuple_processing
pip install hep_ml

git clone git@github.com:boostedsvj/svj_uboost
```

Alternatively, an editable `svj_ntuple_processing` can be installed for simultaneous developments:
```
git clone git@github.com:boostedsvj/svj_ntuple_processing
pip install -e svj_ntuple_processing/
```

## How to run a training

First download the training data (~4.7 Gb), and split it up into a training and test sample:

```bash
python download.py
python split_train_test.py
```

This should give you the following directory structure:

```bash
$ ls data/
bkg  signal  test_bkg  test_signal  train_bkg  train_signal

$ ls data/train_bkg/Summer20UL18/
QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8.npz
QCD_Pt_120to170_TuneCP5_13TeV_pythia8.npz
... <more>
```

Then launch the training script:

```bash
python training.py xgboost \
    --reweight mt --ref data/train_signal/madpt300_mz350_mdark10_rinv0.3.npz \
    --lr .05 \
    --minchildweight .1 \
    --maxdepth 6 \
    --subsample 1. \
    --nest 400
```

Training with xgboost on the full background should take about 45 min.
The script `hyperparameteroptimization.py` runs this command for various settings in parallel.


## Evaluate

```bash
python evaluate.py
```

The paths to the model are currently hard-coded! Things are still too fluid for a good abstraction.


## Produce histograms

To produce the histograms for quick investigation, use:

```bash
python produce_histograms.py models/svjbdt_Nov29_reweight_mt_lr0.05_mcw0.1_maxd6_subs1.0_nest400.json
```

It creates a file called `histograms_%b%d.json`, which contains background and signal histograms for various BDT working points (currently 0.0 to 0.9).

To make some quick debug plots for the histograms:

```bash
python plot_histograms.py histograms_Dec01.json
```


## Cutflow table

```bash
python cutflow_table.py
```

Creates the cutflow tables to inspect the preselection efficiencies. Two example tables (run the script for more):

```
--------------------------------------------------------------------------------
bkg_summary
              qcd      ttjets   wjets    zjets    combined
xs            1.11e+05 1.31e+03 3.32e+03 4.01e+02 1.16e+05
raw           100.00%  100.00%  100.00%  100.00%  100.00%
ak8jet.pt>500 0.99%    0.99%    0.15%    0.27%    0.96%
triggers      0.93%    0.98%    0.15%    0.27%    0.91%
n_ak15jets>=2 0.93%    0.98%    0.15%    0.26%    0.91%
subl_eta<2.4  0.92%    0.98%    0.15%    0.25%    0.90%
subl_ecf>0    0.90%    0.96%    0.12%    0.20%    0.87%
rtx>1.1       0.09%    0.48%    0.06%    0.15%    0.10%
nleptons=0    0.09%    0.20%    0.03%    0.15%    0.09%
metfilter     0.09%    0.20%    0.03%    0.15%    0.09%
preselection  0.09%    0.20%    0.03%    0.15%    0.09%
stitch        0.09%    0.08%    0.03%    0.15%    0.09%
n137          1.39e+07 3.52e+05 1.30e+05 8.28e+04 1.44e+07
--------------------------------------------------------------------------------
signal
              mz250_rinv0.1 mz250_rinv0.3 mz350_rinv0.1 mz350_rinv0.3 mz450_rinv0.1 mz450_rinv0.3
xs            1.14e+02      1.14e+02      9.92e+01      9.92e+01      8.23e+01      8.23e+01     
raw           100.00%       100.00%       100.00%       100.00%       100.00%       100.00%      
ak8jet.pt>500 20.09%        19.06%        22.06%        21.02%        25.01%        23.40%       
triggers      19.78%        18.84%        21.77%        20.80%        24.74%        23.23%       
n_ak15jets>=2 19.78%        18.84%        21.77%        20.80%        24.74%        23.23%       
subl_eta<2.4  19.68%        18.71%        21.68%        20.68%        24.64%        23.12%       
subl_ecf>0    19.36%        18.28%        21.36%        20.22%        24.33%        22.66%       
rtx>1.1       3.95%         10.30%        5.07%         12.52%        6.21%         14.59%       
nleptons=0    3.81%         9.98%         4.87%         12.06%        5.94%         14.00%       
metfilter     3.77%         9.86%         4.80%         11.91%        5.87%         13.83%       
preselection  3.77%         9.86%         4.80%         11.91%        5.87%         13.83%       
stitch        3.77%         9.86%         4.80%         11.91%        5.87%         13.83%       
n137          5.89e+05      1.54e+06      6.53e+05      1.62e+06      6.63e+05      1.56e+06     
```


## Overfitting check: Kolmogorov-Smirnov test

```bash
python overfitting.py models/svjbdt_Nov29_reweight_mt_lr0.05_mcw0.1_maxd6_subs1.0_nest400.json
```

![overfit plot](example_plots/overfit.png)

With p-values close to 1.0, there is no reason to assume any overfitting.


## Scale uncertainties

```
mkdir data/scaleunc
xrdcp root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/signal_madpt300_2023_scaleunc/BDTCOLS/madpt300_mz350_mdark10_rinv0.3_scaleunc.npz data/scaleunc/
python study_scaleunc.py plot data/scaleunc/madpt300_mz350_mdark10_rinv0.3_scaleunc.npz models/svjbdt_Apr21_reweight_mt.json
```

![scale uncertainty plot](example_plots/scaleunc.png)

## Building Data Cards

One of the key files in this repo is the `build_datacard.py` this is the magic file that makes everything come together. Here, datacards that can be fed into combine are built. The first step is to produce npz 'skims' of the signal files while applying the selection (either cutbased or bdt-based). This is done for individual signal samples to calculate all the necessary signal systematics. (The argument `--keep X` can be included to select a random subset of signal events for statistical studies, where `X` is a float between 0 and 1.)

```bash
# For BDT based choose a bdt working point (the DDT is applied while running)
python build_datacard.py skim bdt=0.5 /path/to/signal_file.root 
# For the cut based
python build_datacard.py skim cutbased /path/to/signal_file.root 
```

The background estimation is done by creating function that fits well to background mc and then is applied to data, thus the only uncertainty in the background estimation are the parameters of the fit function. Therefore, no set of 'up and down' histograms are needed for the background mc files. Instead, the selection (bdt or cutbased) is applied to the background samples and a json file with the mT histograms (with user setbinwidth `binw`) is made with the command:
```bash
python3 build_datacard.py build_bkg_histograms --binw 10 cutbased path/to/background/*.npz
```
This step is only done once and reused for all the different signal points.

For signal, the mT histograms including all systematic variations for one signal point should be made with the following command:
```bash
python3 build_datacard.py build_sig_histograms --mtmin 130 --mtmax 700 --binw 10 cutbased path/to/signal/*.npz
```

A larger mT range than the final selection is used to facilitate smoothing of the signal shapes (using local regression), which is performed by:
```bash
python build_datacard.py smooth_shapes --optimize 1000 --target central --mtmin 180 --mtmax 650 signal.json
```
The output histograms from this step are truncated to the final mT range. (`--target central` means that the optimization of the smoothing span via generalized cross-validation uses the central histogram, and then that optimized span value is applied to the systematic variations.)

Finally, the signal and background histograms are combined to make the full input for a datacard:
```bash
python build_datacard.py build_histograms --binw 10 signal_smooth.json bkghist.json
```
The resulting merged file should use the signal name, the selection type, bin widths, and ranges: `signal_name_cutbased_or_bdt_smooth_with_bkg_binwXY_rangeXYZ-XYZ.json`.
(The `build_histograms` function can also be used to perform the individual sig and bkg histogram steps, if it is given npz files instead of json files. In this case, the signal smoothing would have to be performed similarly.)

An additional function for checking the histogram json files is `ls`. However, this is not the most easy to read it provides a quick way to check for mistakes during file creation.

```bash
python3 build_datacard.py ls signal_name_cutbased_or_bdt_with_bkg_binwXY_rangeXYZ-XYZ.json
# or alternatively, just look at it
head -n 100 signal_name_cutbased_or_bdt_with_bkg_binwXY_rangeXYZ-XYZ.json
```

Then all the up, down, and nominal values can be plotted for the systematics:

```bash
python build_datacard.py plot_systematics signal_name_cutbased_or_bdt_with_bkg_binwXY_rangeXYZ-XYZ.json
```

Similar plots can be made to compare the results of smoothing (e.g. for systematics, between different `keep` percentages, etc.) using the `plot_smooth` function.

A table of systematic uncertainty yield effects can be made as follows:
```bash
python build_datacard.py systematics_table signal_name_cutbased_or_bdt_with_bkg_binwXY_rangeXYZ-XYZ.json
```
Currently, this function only handles one signal model at a time.
It will be expanded to summarize across all signal models once the full scans are available.

And that's it for this part. To use these histograms for fits and limit setting, see the [svj_limits](https://github.com/boostedsvj/svj_limits) repo.
