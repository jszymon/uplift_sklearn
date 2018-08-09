"""Hillstrom challenge dataset.

A dataset from Kevin Hillstrom's MineThatData blog. See

    https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

for details.
"""

import logging
from os.path import dirname, exists, join
from os import remove
import csv

import numpy as np

from sklearn.datasets.base import get_data_home
from sklearn.datasets.base import _fetch_remote
from sklearn.datasets.base import RemoteFileMetadata
from sklearn.utils import Bunch
from sklearn.externals import joblib
from sklearn.utils import check_random_state
from sklearn.utils.fixes import makedirs

ARCHIVE = RemoteFileMetadata(
    filename="Hillstrom.csv",
    url=('http://www.minethatdata.com/'
         'Kevin_Hillstrom_MineThatData_E-MailAnalytics'
         '_DataMiningChallenge_2008.03.20.csv'),
    checksum=('0e5893329d8b93cefecc571777672028'
              '290ab69865718020c78c7284f291aece'))

logger = logging.getLogger(__name__)


def fetch_Hillstrom(data_home=None, download_if_missing=True,
                    random_state=None, shuffle=False,
                    categ_as_strings=False, return_X_y=False):
    """Load the Hillstrom dataset (uplift classification and regression).

    Download it if necessary.

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    categ_as_strings : bool, default=False
        Whether to return categorical variables as strings.

    return_X_y : boolean, default=False.
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (581012, 54)
        Each row corresponds to the 54 features in the dataset.

    dataset.target : numpy array of shape (581012,)
        Each value corresponds to one of the 7 forest covertypes with values
        ranging between 1 to 7.

    dataset.DESCR : string
        Description of the forest covertype dataset.

    (data, target) : tuple if ``return_X_y`` is True"""

    data_home = get_data_home(data_home=data_home)
    Hillstrom_dir = join(data_home, "uplift_sklearn", "Hillstrom")
    if categ_as_strings:
        samples_path = join(Hillstrom_dir, "samples_str")
        targets_path = join(Hillstrom_dir, "targets_str")
    else:
        samples_path = join(Hillstrom_dir, "samples")
        targets_path = join(Hillstrom_dir, "targets")
    available = exists(samples_path)

    # dictionaries
    feature_names = ['recency', 'history_segment', 'history', 'mens', 'womens',
                     'zip_code', 'newbie','channel']
    target_names = ["visit", "conversion", "spend"]
    group_values = ['No E-Mail', 'Mens E-Mail', 'Womens E-Mail']
    history_segment_values = ['1) $0 - $100', '2) $100 - $200',
                              '3) $200 - $350', '4) $350 - $500',
                              '5) $500 - $750', '6) $750 - $1,000',
                              '7) $1,000 +']
    zip_code_values = ['Rural', 'Surburban', 'Urban']
    channel_values = ['Phone', 'Web', 'Multichannel']
    categ_values = {"history_segment": history_segment_values,
                    "zip_code": zip_code_values,
                    "channel": channel_values,}

    if download_if_missing and not available:
        if not exists(Hillstrom_dir):
            makedirs(Hillstrom_dir)
        logger.info("Downloading %s" % ARCHIVE.url)

        archive_path = _fetch_remote(ARCHIVE, dirname=Hillstrom_dir)
        #print(archive_path)
        #archive_path = "/home/szymon/scikit_learn_data/usklearn_Hillstrom/Hillstrom.csv"
        # read the data
        Xy = []
        with open(archive_path) as csvfile:
            header = next(csvfile).strip().split(',')
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for record in csvreader:
                Xy.append(record)
                assert len(record) == 12, record
        # delete archive
        remove(archive_path)
        # decode treatment group
        group = [r[8] for r in Xy]
        if categ_as_strings:
            group = np.asarray(group, dtype="U13")
        else:
            group = [group_values.index(g) for g in group]
            group = np.asarray(group, dtype=np.int32)
        # decode targets
        y_visit = np.asarray([int(r[9]) for r in Xy], dtype=np.int32)
        y_conversion = np.asarray([int(r[10]) for r in Xy], dtype=np.int32)
        y_spend = np.asarray([float(r[11]) for r in Xy], dtype=float)
        # decode features
        if categ_as_strings:
            dt = [('recency', int),
                  ('history_segment', 'U16'),
                  ('history', float),
                  ('mens', bool),
                  ('womens', bool),
                  ('zip_code', 'U9'),
                  ('newbie', bool),
                  ('channel', 'U12'),]
        else:
            dt = [('recency', int),
                  ('history_segment', int),
                  ('history', float),
                  ('mens', bool),
                  ('womens', bool),
                  ('zip_code', int),
                  ('newbie', bool),
                  ('channel', int),]
            for r in Xy:
                r[1] = history_segment_values.index(r[1])
                r[5] = zip_code_values.index(r[5])
                r[7] = channel_values.index(r[7])
        X = [tuple(r[:-4]) for r in Xy]
        X = np.asarray(X, dtype=np.dtype(dt))

        joblib.dump(X, samples_path, compress=9)
        joblib.dump((y_visit, y_conversion, y_spend), targets_path, compress=9)

    elif not available and not download_if_missing:
        raise IOError("Data not found and `download_if_missing` is False")
    try:
        X, y_visit, y_conversion, y_spend
    except NameError:
        X = joblib.load(samples_path)
        y_visit, y_conversion, y_spend = joblib.load(targets_path)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        group = group[ind]
        y_visit = y_visit[ind]
        y_conversion = y_conversion[ind]
        y_spend = y_spend[ind]

    #module_path = dirname(__file__)
    #with open(join(module_path, 'descr', 'Hillstrom.rst')) as rst_file:
    #    fdescr = rst_file.read()

    if return_X_y:
        return X, y_visit, y_conversion, y_spend

    return Bunch(data=X, target_visit=y_visit, target_conversion=y_conversion,
                 target_spend=y_spend, categ_values=categ_values,
                 group_values=group_values,
                 feature_names=feature_names, target_names=target_names,
                 DESCR=__doc__)
