import os, os.path as osp, logging, re, time, json, argparse, sys
import matplotlib.pyplot as plt
from collections import OrderedDict
from contextlib import contextmanager
import svj_ntuple_processing
from scipy.ndimage import gaussian_filter
import requests
import numpy as np
np.random.seed(1001)



# Where training data will be stored
DATADIR = osp.join(osp.dirname(osp.abspath(__file__)), 'data')


def setup_logger(name: str = "hadd") -> logging.Logger:
    """Sets up a Logger instance.

    If a logger with `name` already exists, returns the existing logger.

    Args:
        name (str, optional): Name of the logger. Defaults to "demognn".

    Returns:
        logging.Logger: Logger object.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info("Logger %s is already defined", name)
    else:
        fmt = logging.Formatter(
            fmt=(
                "\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m"
                + " %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def debug(flag: bool = True) -> None:
    """Convenience switch to set the logging level to DEBUG.

    Args:
        flag (bool, optional): If true, set the logging level to DEBUG. Otherwise, set
            it to INFO. Defaults to True.
    """
    if flag:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def set_matplotlib_fontsizes(
    small: int = 18, medium: int = 22, large: int = 26
) -> None:
    """Sets matplotlib font sizes to sensible defaults.

    Args:
        small (int, optional): Font size for text, axis titles, and ticks. Defaults to
            18.
        medium (int, optional): Font size for axis labels. Defaults to 22.
        large (int, optional): Font size for figure title. Defaults to 26.
    """
    import matplotlib.pyplot as plt

    plt.rc("font", size=small)  # controls default text sizes
    plt.rc("axes", titlesize=small)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small)  # legend fontsize
    plt.rc("figure", titlesize=large)  # fontsize of the figure title


set_matplotlib_fontsizes()


def pull_arg(*args, **kwargs) -> argparse.Namespace:
    """Reads a specific argument out of sys.argv, and then deletes that argument from
    sys.argv.

    This useful to build very adaptive command line options to scripts. It does
    sacrifice auto documentation of the command line options though.

    Returns:
        argparse.Namespace: Namespace object for only the specific argument.
    """

    """
    Reads a specific argument out of sys.argv, and then
    deletes that argument from sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    return args


@contextmanager
def timeit(msg):
    """
    Prints duration of a block of code in the terminal.
    """
    try:
        logger.info(msg)
        sys.stdout.flush()
        t0 = time.time()
        yield None
    finally:
        t1 = time.time()
        logger.info(f"Done {msg[0].lower() + msg[1:]}, took {t1-t0:.2f} secs")


time_and_log = timeit # backwards compatibility


def imgcat(path) -> None:
    """
    Only useful if you're using iTerm with imgcat on the $PATH:
    Display the image in the terminal.
    """
    try:
        os.system('imgcat ' + path)
    except Exception:
        pass


class Scripter:
    """
    Command line utility.

    When an instance of this class is used as a contextwrapper on a function, that
    function will be considered a 'command'.

    When Scripter.run() is called, the script name is pulled from the command line, and
    the corresponding function is executed.

    Example:

        In file test.py:
        >>> scripter = Scripter()
        >>> @scripter
        >>> def my_func():
        >>>     print('Hello world!')
        >>> scripter.run()

        On the command line, the following would print 'Hello world!':
        $ python test.py my_func
    """

    def __init__(self):
        self.scripts = {}

    def __call__(self, fn):
        """
        Stores a command line script with its name as the key.
        """
        self.scripts[fn.__name__] = fn
        return fn

    def run(self):
        script = pull_arg("script", choices=list(self.scripts.keys())).script
        logger.info(f"Running {script}")
        self.scripts[script]()


@contextmanager
def quick_ax(figsize=(10, 10), outfile="tmp.png"):
    """
    Context manager to open a matplotlib Axes.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        yield ax
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass


@contextmanager
def quick_fig(figsize=(10, 10), outfile="tmp.png"):
    """
    Context manager to open a matplotlib Figure.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        yield fig
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass


#__________________________________________________
# Automatic cross section getter

class Record(dict):
    @property
    def xs(self):
        return self['crosssection']['xs_13tev']

    @property
    def br(self):
        try:
            return self['branchingratio']['br_13tev']
        except KeyError:
            return 1.

    @property
    def kfactor(self):
        if 'kfactor' in self:
            for key, val in self['kfactor'].items():
                if key.startswith('kfactor_'):
                    return val
        return 1.

    @property
    def effxs(self):
        return self.xs*self.br*self.kfactor


def load_treemaker_crosssection_txt():
    """
    Downloads the cross section file from the TreeMaker repository and returns
    the contents. If the file has been previously downloaded it is not re-downloaded.
    """
    import requests
    cache = '/tmp/treemaker_xs.txt'
    if not osp.isfile('/tmp/treemaker_xs.txt'):
        url = 'https://raw.githubusercontent.com/TreeMaker/TreeMaker/Run2_UL/WeightProducer/python/MCSampleValues.py'
        text = requests.get(url).text
        with open(cache, 'w') as f:
            text = text.lower()
            f.write(text)
            return text
    else:
        with open(cache) as f:
            return f.read()


def get_record(key):
    """
    Looks for the sample key (e.g. "QCD_Pt_1400to1800") in the cross section
    file from TreeMaker
    """
    text = load_treemaker_crosssection_txt()
    match = re.search('"'+key+'"' + r' : ({[\w\W]*?})', text, re.MULTILINE)
    if not match: raise Exception(f'Could not find record for {key}')
    # Turn it into a dict of dicts
    record_txt = (
        match.group(1)
        .replace('xsvalues', 'dict')
        .replace('brvalues', 'dict')
        .replace('kfactorvalues', 'dict')
        )
    return Record(eval(record_txt))


def mt_wind(cols, mt_high, mt_low):
    mt_cut = (cols.arrays['mt']>mt_low) & (cols.arrays['mt']<mt_high)
    return mt_cut

def filter_pt(cols, min_pt):
    """
    Filters for a minimum pt (only valid for QCD).
    Does not filter out non-QCD backgrounds or signals.
    """
    filtered = []
    for c in cols:
        if c.metadata.get('ptbin', [1e6])[0] < min_pt:
            continue
        filtered.append(c)
    return filtered

def filter_ht(cols, min_ht, bkg_type=None):
    """
    Filters for a minimum pt (only valid for ttjets/wjets/zjets).
    Does not filter out QCD or signal.
    If bkg_type is None, it filters ttjets AND wjets AND zjets.
    """
    filtered = []
    for c in cols:
        if bkg_type and c.metadata.get('bkg_type', None) != bkg_type:
            filtered.append(c)
            continue
        if c.metadata.get('htbin',[1e6, 1e6])[0] < min_ht:
            continue
        filtered.append(c)
    return filtered


#__________________________________________________
# Data pipeline

class Columns(svj_ntuple_processing.Columns):
    """
    Data structure that contains all the training data (features)
    and information about the sample.
    
    See: https://github.com/boostedsvj/svj_ntuple_processing/blob/main/svj_ntuple_processing/__init__.py#L357
    """
    @classmethod
    def load(cls, *args, **kwargs):
        inst = super().load(*args, **kwargs)
        # Transforming bytes keys to str keys
        old_cf = inst.cutflow
        inst.cutflow = OrderedDict()
        for key in old_cf.keys():
            if isinstance(key, bytes):
                inst.cutflow[key.decode()] = old_cf[key]
            else:
                inst.cutflow[key] = old_cf[key]
        return inst

    def __repr__(self):
        return (
            '<Column '
            + ' '.join([f'{k}={v}' for k, v in self.metadata.items() if k!='src'])
            + '>'
            )

    @property
    def key(self):
        return (
            osp.basename(self.metadata['src'])
            .replace('.npz', '')
            ).split('_TuneCP5_13TeV')[0].lower()

    @property
    def record(self):
        if not hasattr(self, '_record'):
            self._record = get_record(self.key)
        return self._record

    @property
    def presel_eff(self):
        if self.cutflow['raw']==0: return 0.
        return self.cutflow['preselection'] / self.cutflow['raw']

    @property
    def xs(self):
        if hasattr(self, 'manual_xs'):
            # Only for setting a manual cross section
            return self.manual_xs
        elif 'bkg_type' in self.metadata:
            return self.record.effxs
        else:
            if not hasattr(self, '_signal_xs_fit'):
                self._signal_xs_fit = np.poly1d(
                    requests
                    .get('https://raw.githubusercontent.com/boostedsvj/svj_madpt_crosssection/main/fit_madpt300.txt')
                    .json()
                    )
            return self._signal_xs_fit(self.metadata['mz'])

    @property
    def effxs(self):
        return self.xs * self.presel_eff

    @property
    def weight_per_event(self):
        return self.effxs / len(self)

def columns_to_numpy_for_training(
    signal_cols, qcd_cols, tt_cols, features,
    downsample=.4, weight_key='weight',
    mt_high=650, mt_low=180
    ):
    """
    Takes a list of signal and background Column instances, and outputs
    a numpy array with `features` as the columns.
    Ensures that ttjets and qcd are normalized to the same number when making weights
    """
    X = []
    y = []
    bkg_weight = []
    tt_weight = []
    signal_weight = []

    logger.info(f'Downsampling bkg, keeping fraction of {downsample}')

    # user defined normalization value
    k = 1000000.0

    # Get the features for the bkg samples
    for cols in qcd_cols:

        # Apply the mt window
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]

        # make sure pile up weights are applied
        if weight_key == 'weight' :
            this_weight = cols.arrays['puweight'][mtwind]*cols.arrays['weight'][mtwind]
        else :
            this_weight = cols.arrays[weight_key][mtwind]

        # apply the down sampling
        if downsample < 1.:
            select = np.random.choice(len(this_weight), int(downsample*len(this_weight)), replace=False)
            this_X = this_X[select]
            this_weight = this_weight[select]
        X.append(this_X)
        bkg_weight.append(this_weight)
        y.append(np.zeros(len(this_X)))

    # concatenate the QCD weights then prepare to add tt weights
    # while ensuring that the QCD weights are weighted to k where k is user input
    bkg_weight = np.concatenate(bkg_weight)
    total_weight = sum(bkg_weight)
    bkg_weight = [weight * (k / total_weight) for weight in bkg_weight]

    # Get the features for the bkg samples
    for cols in tt_cols:

        # Apply the mt window
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]

        # make sure pile up weights are applied
        if weight_key == 'weight' :
            this_weight = cols.arrays['puweight'][mtwind]*cols.arrays['weight'][mtwind]
        else :
            this_weight = cols.arrays[weight_key][mtwind]
        # apply the down sampling
        if downsample < 1.:
            select = np.random.choice(len(this_weight), int(downsample*len(this_weight)), replace=False)
            this_X = this_X[select]
            this_weight = this_weight[select]
        X.append(this_X)
        tt_weight.append(this_weight)
        y.append(np.zeros(len(this_X)))

    # concatenate the tt weights then prepare to append them to the QCD weights
    # while ensuring that the tt weights are weighted to k where k is user input
    tt_weight = np.concatenate(tt_weight)
    #total_weight = sum(tt_weight)
    #tt_weight = [weight * (k / total_weight) for weight in tt_weight] # Investigate
    tt_weight *= np.sum(bkg_weight) / np.sum(tt_weight) # not sure if necessary INVESTIGATE
    bkg_weight = np.concatenate((bkg_weight, tt_weight))

    # Get the features for the signal samples
    for cols in signal_cols:
        sigmtwind = mt_wind(cols, mt_high, mt_low)
        X.append(cols.to_numpy(features)[sigmtwind])
        len_sig_cols=len(cols.arrays[features[0]][sigmtwind])
        y.append(np.ones(len_sig_cols))
        # All signal model parameter variations should get equal weight,
        # but some signal samples have more events.
        # Use 1/n_events as a weight per event.
        signal_weight.append((1./len_sig_cols)*np.ones(len_sig_cols))
    
    signal_weight = np.concatenate(signal_weight)
    # Set total signal weight equal to total bkg weight
    signal_weight *= np.sum(bkg_weight) / np.sum(signal_weight)
    weight = np.concatenate((bkg_weight, signal_weight))

    print(weight)

    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y, weight

def columns_to_numpy_for_iter_training(
    signal_cols, qcd_cols, tt_cols, features,
    weight_key='weight',
    mt_high=650, mt_low=180
    ):
    """
    Takes a list of signal and background Column instances, and outputs
    a numpy array with `features` as the columns.
    Ensures that ttjets and qcd are normalized to the same number when making weights
    """
    X = []
    y = []
    bkg_weight = []
    qcd_weight = []
    signal_weight = []

    # user defined normalization value
    k = 1000000.0

    # make sure that the masses don't go outside the 'full' window
    if mt_high > 650: mt_high = 650
    if mt_low < 180: mt_low = 180

    # Make the tt weight for the full mt range
    full_tt_weight = []
    tt_masses = []
    for cols in tt_cols:

        # Apply the mt window
        mtwind = mt_wind(cols, 650.0, 180.0)

        # make sure pile up weights are applied
        this_mass = cols.arrays['mt'][mtwind]
        if weight_key == 'weight' :
            this_weight = cols.arrays['puweight'][mtwind]*cols.arrays['weight'][mtwind]
        else :
            this_weight = cols.arrays[weight_key][mtwind]

        tt_masses.append(this_mass)
        full_tt_weight.append(this_weight)

    # Normalize the tt weights to k where k is user input
    tt_masses = np.concatenate(tt_masses)
    full_tt_weight = np.concatenate(full_tt_weight)
    total_weight = sum(full_tt_weight)
    full_tt_weight = [weight * (k / total_weight) for weight in full_tt_weight]

    #print("full tt weight", full_tt_weight)

    # Get the features for the bkg samples
    for cols in tt_cols:

        # Apply the mt window
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]
        X.append(this_X)
        y.append(np.zeros(len(this_X)))

    tt = np.concatenate(X)

    # concatenate the QCD weights then prepare to add tt weights
    # while ensuring that the QCD weights are weighted to k where k is user input
    full_tt_weight = np.array(full_tt_weight)
    tt_mask = (tt_masses > mt_low) & (tt_masses < mt_high)
    bkg_weight = full_tt_weight[tt_mask]

    # Get the features for the bkg samples
    for cols in qcd_cols:

        # Apply the mt window
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]

        # make sure pile up weights are applied
        if weight_key == 'weight' :
            this_weight = cols.arrays['puweight'][mtwind]*cols.arrays['weight'][mtwind]
        else :
            this_weight = cols.arrays[weight_key][mtwind]

        X.append(this_X)
        qcd_weight.append(this_weight)
        y.append(np.zeros(len(this_X)))

    # concatenate the QCD weights then prepare to append them to the tt weights
    # while ensuring that QCD is weighted so that it i equal to the tt sample
    qcd_weight = np.concatenate(qcd_weight)
    qcd_weight *= np.sum(bkg_weight) / np.sum(qcd_weight) # not sure if necessary INVESTIGATE
    bkg_weight = np.concatenate((bkg_weight, qcd_weight))

    # Get the features for the signal samples
    for cols in signal_cols:
        sigmtwind = mt_wind(cols, mt_high, mt_low)
        X.append(cols.to_numpy(features)[sigmtwind])
        len_sig_cols=len(cols.arrays[features[0]][sigmtwind])
        y.append(np.ones(len_sig_cols))
        # All signal model parameter variations should get equal weight,
        # but some signal samples have more events.
        # Use 1/n_events as a weight per event.
        signal_weight.append((1./len_sig_cols)*np.ones(len_sig_cols))
    
    signal_weight = np.concatenate(signal_weight)
    # Set total signal weight equal to total bkg weight
    signal_weight *= np.sum(bkg_weight) / np.sum(signal_weight)
    weight = np.concatenate((bkg_weight, signal_weight))

    #print("resulting weights", weight)

    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y, weight

def columns_to_numpy_one_bkg(
    signal_cols, bkg_cols, features,
    downsample=.4, weight_key='weight',
    mt_high=650, mt_low=180
    ):
    """
    Takes a list of signal and background Column instances, and outputs
    a numpy array with `features` as the columns.
    This is only to be used with qcd OR tt jets; it assums one bkg type
    """
    X = []
    y = []
    bkg_weight = []
    signal_weight = []

    logger.info(f'Downsampling bkg, keeping fraction of {downsample}')

    # user defined normalization value
    k = 1000000.0

    # Get the features for the bkg samples
    for cols in bkg_cols:

        # Apply the mt window
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]

        # make sure pile up weights are applied
        if weight_key == 'weight' :
            this_weight = cols.arrays['puweight'][mtwind]*cols.arrays['weight'][mtwind]
        else :
            this_weight = cols.arrays[weight_key][mtwind]

        # apply the down sampling
        if downsample < 1.:
            select = np.random.choice(len(this_weight), int(downsample*len(this_weight)), replace=False)
            this_X = this_X[select]
            this_weight = this_weight[select]
        X.append(this_X)
        bkg_weight.append(this_weight)
        y.append(np.zeros(len(this_X)))

    # concatenate the QCD weights then prepare to add tt weights
    # while ensuring that the QCD weights are weighted to k where k is user input
    bkg_weight = np.concatenate(bkg_weight)
    total_weight = sum(bkg_weight)
    bkg_weight = [weight * (k / total_weight) for weight in bkg_weight]

    # Get the features for the signal samples
    for cols in signal_cols:
        sigmtwind = mt_wind(cols, mt_high, mt_low)
        X.append(cols.to_numpy(features)[sigmtwind])
        len_sig_cols=len(cols.arrays[features[0]][sigmtwind])
        y.append(np.ones(len_sig_cols))
        # All signal model parameter variations should get equal weight,
        # but some signal samples have more events.
        # Use 1/n_events as a weight per event.
        signal_weight.append((1./len_sig_cols)*np.ones(len_sig_cols))
    
    signal_weight = np.concatenate(signal_weight)
    # Set total signal weight equal to total bkg weight
    signal_weight *= np.sum(bkg_weight) / np.sum(signal_weight)
    weight = np.concatenate((bkg_weight, signal_weight))

    print(weight)

    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y, weight




def columns_to_numpy(
    signal_cols, bkg_cols, features,
    downsample=.4, weight_key='weight',
    mt_high=650, mt_low=180
    ):
    """
    Takes a list of signal and background Column instances, and outputs
    a numpy array with `features` as the columns.
    """
    X = []
    y = []
    bkg_weight = []
    signal_weight = []

    logger.info(f'Downsampling bkg, keeping fraction of {downsample}')

    # Get the features for the bkg samples
    for cols in bkg_cols:

        # Apply the mt window
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]

        # make sure pile up weights are applied
        if weight_key == 'weight' :
            this_weight = cols.arrays['puweight'][mtwind]*cols.arrays['weight'][mtwind]
        else :
            this_weight = cols.arrays[weight_key][mtwind]
        #len_bkg_cols = len(cols.arrays[weight_key][mtwind])
        #print(len_bkg_cols)
        #this_weight = (1./len_bkg_cols)*np.ones(len_bkg_cols)
        #this_weight = (1./len_bkg_cols)*np.ones(len_bkg_cols)*(0.000001)

        # apply the down sampling
        if downsample < 1.:
            #select = np.random.choice(len(cols), int(downsample*len(cols)), replace=False)
            select = np.random.choice(len(this_weight), int(downsample*len(this_weight)), replace=False)
            this_X = this_X[select]
            this_weight = this_weight[select]
        X.append(this_X)
        bkg_weight.append(this_weight)
        y.append(np.zeros(len(this_X)))

    # Get the features for the signal samples
    for cols in signal_cols:
        sigmtwind = mt_wind(cols, mt_high, mt_low)
        X.append(cols.to_numpy(features)[sigmtwind])
        #print(features)
        len_sig_cols=len(cols.arrays[features[0]][sigmtwind])
        #print(cols.to_numpy(features)[sigmtwind])
        #print(len(cols.to_numpy(features)[sigmtwind]))
        #length_of_signalCol=len(cols.arrays(features)[mtwind])
        #print(length_of_signalCol, len(cols))
        y.append(np.ones(len_sig_cols))
        # All signal model parameter variations should get equal weight,
        # but some signal samples have more events.
        # Use 1/n_events as a weight per event.
        signal_weight.append((1./len_sig_cols)*np.ones(len_sig_cols))
    
    bkg_weight = np.concatenate(bkg_weight)
    signal_weight = np.concatenate(signal_weight)
    # Set total signal weight equal to total bkg weight
    signal_weight *= np.sum(bkg_weight) / np.sum(signal_weight)
    weight = np.concatenate((bkg_weight, signal_weight))

    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y, weight

def columns_to_numpy_single(
    cols, features,
    mt_high=650, mt_low=180
    ):
    """
    Takes a list of signal and background Column instances, and outputs
    a numpy array with `features` as the columns.
    """
    X = []

    # Get the features for the bkg samples
    for col in cols:

        # Apply the mt window
        mtwind = mt_wind(col, mt_high, mt_low)
        this_X = col.to_numpy(features)[mtwind]
        X.append(this_X)

    X = np.concatenate(X)
    return X

def add_key_value_to_json(json_file, key, value):
    with open(json_file, 'r') as f:
        json_str = f.read()
    json_str = json_str.rstrip()
    json_str = json_str[:-1] # Strip off the last }
    json_str += f',"{key}":{json.dumps(value)}}}'
    with open(json_file, 'w') as f:
        f.write(json_str)
    logger.info(f'Added "{key}":{json.dumps(value)} to {json_file}')


def add_manual_weight_column(signal_cols, bkg_cols):
    """
    Adds the manual weight calculation as a column
    """
    total_bkg_weight = 0
    for c in bkg_cols:
        c.arrays['manualweight'] = np.ones(len(c)) * c.weight_per_event
        total_bkg_weight += c.arrays['manualweight'].sum()

    # Set signal weights scaled relatively to one another (but not yet w.r.t. bkg)
    total_signal_weight = 0
    for c in signal_cols:
        c.arrays['manualweight'] = np.ones(len(c)) / len(c)
        total_signal_weight += c.arrays['manualweight'].sum()
    
    # Scale signal weights correctly w.r.t. bkg
    for c in signal_cols:
        c.arrays['manualweight'] *= total_bkg_weight / total_signal_weight


def read_training_features(model_file):
    """
    Reads the features used to train a model from a .json file.
    Only works for xgboost-trained models.
    """
    with open(model_file, 'rb') as f:
        model = json.load(f)
        return model['features']

def rhoddt_windowcuts(mt, pt, rho):
    cuts = (mt>200) & (mt<1000) & (pt>110) & (pt<1500) & (rho>-4) & (rho<0)
    return cuts

def varmap(mt, pt, rho, var, weight):
    cuts = rhoddt_windowcuts(mt, pt, rho)
    C, RHO_edges, PT_edges = np.histogram2d(rho[cuts], pt[cuts], bins=49,weights=weight[cuts])
    w, h = 50, 50
    VAR_map      = [[0 for x in range(w)] for y in range(h)]
    VAR = var[cuts]
    for i in range(len(RHO_edges)-1):
       for j in range(len(PT_edges)-1):
          CUT = (rho[cuts]>RHO_edges[i]) & (rho[cuts]<RHO_edges[i+1]) & (pt[cuts]>PT_edges[j]) & (pt[cuts]<PT_edges[j+1])
          if len(VAR[CUT])==0: continue
          if len(VAR[CUT])>0:
             #VAR_map[i][j]=np.percentile(VAR[CUT],18.2) # bdt>0.6
             VAR_map[i][j]=np.percentile(VAR[CUT],36.2) # bdt>0.4

    VAR_map_smooth = gaussian_filter(VAR_map,1)
    return VAR_map_smooth, RHO_edges, PT_edges




def ddt(mt, pt, rho, var, weight):
    cuts = rhoddt_windowcuts(mt, pt, rho)
    var_map_smooth, RHO_edges, PT_edges = varmap(mt, pt, rho, var, weight)
    nbins = 49
    Pt_min, Pt_max = min(PT_edges), max(PT_edges)
    Rho_min, Rho_max = min(RHO_edges), max(RHO_edges)

    ptbin_float  = nbins*(pt-Pt_min)/(Pt_max-Pt_min) 
    rhobin_float = nbins*(rho-Rho_min)/(Rho_max-Rho_min)

    #ptbin         = np.clip(1 + ptbin_float.astype(int),   0, nbins)
    #rhobin        = np.clip(1 + rhobin_float.astype(int),  0, nbins)
    ptbin  = np.clip(1 + np.round(ptbin_float).astype(int), 0, nbins)
    rhobin = np.clip(1 + np.round(rhobin_float).astype(int), 0, nbins)

    varDDT = np.array([var[i] - var_map_smooth[rhobin[i]-1][ptbin[i]-1] for i in range(len(var))])
    return varDDT
