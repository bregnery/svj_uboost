import os, os.path as osp, logging, re, time, json, argparse, sys, math, shutil
import matplotlib.pyplot as plt
from collections import OrderedDict
from contextlib import contextmanager
import svj_ntuple_processing
from scipy.ndimage import gaussian_filter
import requests
import numpy as np
from datetime import datetime
import json

np.random.seed(1001)


# Is this a good binning?
MT_BINS = np.linspace(100., 1000., 101)

# Where training data will be stored
DATADIR = osp.join(osp.dirname(osp.abspath(__file__)), 'data')


def setup_logger(name: str = "bdt") -> logging.Logger:
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


cms_style = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    # 
    "mathtext.fontset": "custom",
    "mathtext.rm": "helvetica",
    "mathtext.bf": "helvetica:bold",
    "mathtext.sf": "helvetica",
    "mathtext.it": "helvetica:italic",
    "mathtext.tt": "helvetica",
    "mathtext.cal": "helvetica",
    # 
    "figure.figsize": (10.0, 10.0),
    "font.size": 26,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "savefig.transparent": False,
    "xaxis.labellocation": "right",
    "yaxis.labellocation": "top",
    'text.usetex' : True,    
    }

def set_mpl_fontsize(small=16, medium=22, large=26):
    """Sets matplotlib font sizes to sensible defaults.

    Args:
        small (int, optional): Font size for text, axis titles, and ticks. Defaults to
            18.
        medium (int, optional): Font size for axis labels. Defaults to 22.
        large (int, optional): Font size for figure title. Defaults to 26.
    """
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)    # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title
    from matplotlib.pyplot import style as plt_style
    plt_style.use(cms_style)
    plt.rc('text', usetex=True)
    plt.rc(
        'text.latex',
        preamble=(
            r'\usepackage{helvet} '
            r'\usepackage{sansmath} '
            r'\sansmath '
            )
        )

def put_on_cmslabel(ax, text='Simulation Preliminary', year=2018):
    fontsize = 27
    ax.text(
        .0, 1.005,
        r'\textbf{CMS}\,\fontsize{21pt}{3em}\selectfont{}{\textit{'+text+'}}',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=fontsize
        )
    ax.text(
        1.0, 1.005,
        '{} (13 TeV)'.format(year),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=int(19./23. * fontsize)
        )

import matplotlib.pyplot as plt
set_mpl_fontsize()



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
    if shutil.which('imgcat'):
        os.system('imgcat ' + path)


def expand_wildcards(pats):
    import seutils
    expanded = []
    for pat in pats:
        if '*' in pat:
            if seutils.path.has_protocol(pat):
                expanded.extend(seutils.ls_wildcard(pat))
            else:
                expanded.extend(glob.glob(pat))
        else:
            expanded.append(pat)
    return expanded


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

@contextmanager
def quick_subplots(*args, **kwargs):
    """
    Context manager to open a matplotlib Figure.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    outfile = kwargs.pop('outfile', 'tmp.png')
    try:
        fig, axes = plt.subplots(*args, **kwargs)
        yield fig, axes
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
# Histogram classes

class Histogram:
    """
    Histogram container class.

    Keeps track of binning, values, errors, and metadata.
    Designed to be easily JSON-serializable.
    """
    @classmethod
    def from_dict(cls, dict):
        inst = cls.__new__(cls)
        inst.binning = np.array(dict['binning'])
        inst.vals = np.array(dict['vals'])
        inst.errs = np.array(dict['errs'])
        inst.metadata = dict['metadata'].copy()
        return inst

    def __init__(self, binning, vals=None, errs=None):
        self.binning = binning
        self.vals = np.zeros(self.nbins) if vals is None else vals
        self.errs = np.sqrt(self.vals) if errs is None else errs
        self.metadata = {}

    @property
    def nbins(self):
        return len(self.binning)-1

    def json(self):
        # Convert anything that remotely looks like a float to python float.
        for k, v in self.metadata.items():
            try:
                self.metadata[k] = float(v)
            except ValueError:
                pass
        return dict(
            type = 'Histogram',
            binning = list(self.binning),
            vals = list(self.vals),
            errs = list(self.errs),
            metadata = self.metadata.copy()
            )

    def __repr__(self):
        d = np.column_stack((self.vals, self.errs))
        return (
            f'<H n={self.nbins} int={self.vals.sum():.3f}'
            f' binning={self.binning[0]:.1f}-{self.binning[-1]:.1f}'
            f' vals/errs=\n{d}'
            '>'
            )

    def copy(self):
        the_copy = Histogram(self.binning.copy(), self.vals.copy(), self.errs.copy())
        the_copy.metadata = self.metadata.copy()
        return the_copy

    def __add__(self, other):
        """Add another Histogram or a numpy array to this histogram. Returns new object."""
        ans = self.copy()
        if isinstance(other, Histogram):
            ans.vals = self.vals + other.vals
            ans.errs = np.sqrt(self.errs**2 + other.errs**2)
        elif hasattr(other, 'shape') and self.vals.shape == other.shape:
            # Add a simple np histogram on top of it
            ans.vals += other
            ans.errs = np.sqrt(self.errs**2 + other)
        return ans

    def __radd__(self, other):
        if other == 0:
            return self.copy()
        raise NotImplemented

    def __mul__(self, factor):
        """Multiply by a constant"""
        ans = self.copy()
        if isinstance(factor, (int, float)):
            ans.vals = factor*ans.vals
            ans.errs = factor*ans.errs
        else:
            raise NotImplemented
        return ans

    @property
    def norm(self):
        return self.vals.sum()
    
    def rebin(self, n=2):
        """
        Merge n bins together to make a coarser histogram.
        Mostly useful for plotting.
        """
        if n==1: return self.copy()
        n_bins_new = math.ceil(self.nbins / float(n))

        binning_new = self.binning[::n]
        if binning_new[-1] != self.binning[-1]:
            binning_new = np.append(binning_new, self.binning[-1])

        # Build a map from old binning to new binning
        map = np.repeat(np.arange(n_bins_new), n)[:self.nbins]

        values_new = np.zeros(n_bins_new)
        np.add.at(values_new, map, self.vals)

        errs_new = np.zeros(n_bins_new)
        np.add.at(errs_new, map, self.errs**2)
        errs_new = np.sqrt(errs_new)

        h = Histogram(binning_new, values_new, errs_new)
        h.metadata = self.metadata.copy()
        return h

    def cut(self, xmin=-np.inf, xmax=np.inf):
        """
        Throws away all bins with left boundary < xmin or right boundary > xmax.
        Mostly useful for plotting purposes.
        Returns a copy.
        """
        # safety checks
        if xmin>xmax:
            raise ValueError("xmin ({}) greater than xmax ({})".format(xmin,xmax))

        h = self.copy()
        imin = np.argmin(self.binning < xmin) if xmin > self.binning[0] else 0
        imax = np.argmax(self.binning > xmax) if xmax < self.binning[-1] else self.nbins+1
        h.binning = h.binning[imin:imax]
        h.vals = h.vals[imin:imax-1]
        h.errs = h.errs[imin:imax-1]
        return h


class MTHistogram(Histogram):
    """
    Small wrapper around Histogram that initializes from mt values and weights.
    """

    bins = MT_BINS.copy()
    non_standard_binning = False

    @classmethod
    def empty(cls):
        return Histogram(cls.bins)

    def __init__(self, mt, weights=None):
        vals = np.histogram(mt, self.bins, weights=weights)[0].astype(float)
        errs = np.sqrt(np.histogram(mt, self.bins, weights=weights**2)[0].astype(float))
        super().__init__(self.bins, vals, errs)


class Encoder(json.JSONEncoder):
    """
    Standard JSON encoder, but support for the Histogram class
    """
    def default(self, obj):
        if isinstance(obj, Histogram):
            return obj.json()
        return super().default(obj)


class Decoder(json.JSONDecoder):
    """
    Standard JSON decoder, but support for the Histogram class
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        try:
            is_histogram = d['type'] == 'Histogram'
        except (AttributeError, KeyError):
            is_histogram = False
        if is_histogram:
            return Histogram.from_dict(d)
        return d

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
        mtwind = mt_wind(cols, mt_high, mt_low)
        this_X = cols.to_numpy(features)[mtwind]
        this_weight = cols.arrays[weight_key][mtwind]
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
    '''
    Basically a tool to constantly check the kinematics during the DDT processes
    '''
    cuts = (mt>180) & (mt<650) & (pt>110) & (pt<1500) & (rho>-4) & (rho<0)
    return cuts

def varmap(mt, pt, rho, var, weight, percent):
    '''
    2D map that basically is the DDT
    It decorrelates var with respect to mt using a 2D in pt rho space for a given efficiency (percent)
    '''
    # Apply the rho-ddt window cuts to the data
    cuts = rhoddt_windowcuts(mt, pt, rho)

    # Create a 2D histogram of rho and pt, weighted by the event weights
    C, RHO_edges, PT_edges = np.histogram2d(rho[cuts], pt[cuts], bins=49,weights=weight[cuts])

    # Initialize a 2D map for the variable
    w, h = 50, 50
    VAR_map = [[0 for x in range(w)] for y in range(h)]

    # Get the variable for the data that passed the cuts
    VAR = var[cuts]

    # Loop over the bins in rho and pt
    for i in range(len(RHO_edges)-1):
        for j in range(len(PT_edges)-1):
            # Apply cuts to select data in the current rho and pt bin
            CUT = (rho[cuts]>RHO_edges[i]) & (rho[cuts]<RHO_edges[i+1]) & (pt[cuts]>PT_edges[j]) & (pt[cuts]<PT_edges[j+1])

            # If there is no data in this bin, skip it
            if len(VAR[CUT])==0: continue

            # If there is data in this bin, calculate the percentile of the variable
            if len(VAR[CUT])>0:
                VAR_map[i][j]=np.percentile(VAR[CUT],100-percent) # percent is calculated based on the bdt working point

    # Smooth the variable map using a Gaussian filter
    VAR_map_smooth = gaussian_filter(VAR_map,1)

    # Return the smoothed variable map, along with the rho and pt edges
    return VAR_map_smooth, RHO_edges, PT_edges

# Class that converts numpy arrays into list so they can be easily stored in json files
class NumpyArrayEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)

def create_DDT_map_dict(mt, pt, rho, var, weight, percents, cut_vals, ddt_name):
    '''
    This function creates the dictionary of DDT 2D maps for a range of
        cut_vals at corresponding bkg efficiencies given as percents
    The DDT 2D map at each cut_val contains: var_map_smooth, RHO_edges, and PT_edges, 
        The inputs to the function are (mt, pt, rho, var, weight, percents, cut_vals, ddt_name).
    In essense the DDT is a 2D function of rho and pt that
        is calculated using the background data for the variable
        that you want to decorrelate
    '''

    # Apply the rho-ddt window cuts to the data
    cuts = rhoddt_windowcuts(mt, pt, rho)

    # Initialize the dictionary
    var_dict = {}

    # Loop over the values in cut_vals and percents
    for cut_val, percent in zip(cut_vals, percents):

        print("Creating DDT 2D map for a cut value of ", cut_val, " and an efficiency (percent) of ", percent)

        # Generate a smoothed variable map, along with the rho and pt edges
        var_map_smooth, RHO_edges, PT_edges = varmap(mt, pt, rho, var, weight, percent)

        # Store the results in the dictionary
        var_dict[str(cut_val)] = (var_map_smooth, RHO_edges, PT_edges)

    # Save the dictionary to an json file
    if ddt_name is None:
        ddt_name = 'ddt_' + str(var) + '_' + datetime.now().strftime('%Y%m%d') + '.json'
    with open(ddt_name, 'w') as f:
        json.dump(var_dict, f, cls=NumpyArrayEncoder)

def calculate_varDDT(mt, pt, rho, var, weight, cut_val, ddt_name):
    '''
    This is a function to apply a design decorrelated tagger 
        it decorrelates 'var' with respect to mt using 
        rho (a function of mass) and pt. At a given cut_val for
        a given DDT map inside of an npz file with the name 'ddt_name'
    The inputs to the function are (mt, pt, rho, var, weight, cut_val) 
        where cut_val will refer to the value of the key inside of dictionary 
        with the proper var_map_smooth, RHO_edges, and PT_edges to use.
    '''

    # Check if ddt_name exists
    if not osp.exists(ddt_name):
        raise FileNotFoundError(f"The file {ddt_name} does not exist.")

    # Load the dictionary from the npz file
    with open(ddt_name, 'r') as f:
        var_dict = json.load(f)

    # Check if cut_val exists in the dictionary
    if str(cut_val) not in var_dict:
        raise KeyError(f"The key {cut_val} does not exist in the dictionary.")

    # Get the var_map_smooth, RHO_edges, and PT_edges for the given cut_val
    var_map_smooth, RHO_edges, PT_edges = var_dict[str(cut_val)]
    var_map_smooth = np.array(var_map_smooth)
    RHO_edges = np.array(RHO_edges)
    PT_edges = np.array(PT_edges)

    # Apply the rho-ddt window cuts to the data
    cuts = rhoddt_windowcuts(mt, pt, rho)

    # Define the number of bins and the min/max values for pt and rho
    nbins = 49
    Pt_min, Pt_max = min(PT_edges), max(PT_edges)
    Rho_min, Rho_max = min(RHO_edges), max(RHO_edges)

    # Calculate the floating point bin indices for pt and rho
    ptbin_float  = nbins*(pt-Pt_min)/(Pt_max-Pt_min) 
    rhobin_float = nbins*(rho-Rho_min)/(Rho_max-Rho_min)

    # Convert the floating point bin indices to integer, and clip them to the range [0, nbins]
    ptbin  = np.clip(1 + np.round(ptbin_float).astype(int), 0, nbins)
    rhobin = np.clip(1 + np.round(rhobin_float).astype(int), 0, nbins)

    # Calculate the DDT-transformed variable by subtracting the 
    # decorelation function (smoothed variable map) from the original variable
    varDDT = np.array([var[i] - var_map_smooth[rhobin[i]-1][ptbin[i]-1] for i in range(len(var))])

    # Return the DDT-transformed variable
    return varDDT

def mask_cutbased(col):
    return ((col.arrays['rt'] > 1.18) & (col.arrays['ecfm2b1'] > 0.09))

class InvalidSelectionException(Exception):
    def __init__(self, msg='selection argument should be "cutbased" or "bdt=X.XXX".', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
