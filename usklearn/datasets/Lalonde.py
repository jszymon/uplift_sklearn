"""The Lalonde datasets.

There are two versions of the dataset, including 1974 earnings
(version B) and excluding 1974 earnings (vestion A).
"""


import logging
from os.path import dirname, exists, join
from os import remove, makedirs
import csv

import numpy as np

from sklearn.datasets import get_data_home
from sklearn.datasets._base import _fetch_remote
from sklearn.datasets._base import RemoteFileMetadata
from sklearn.utils import Bunch
import joblib
from sklearn.utils import check_random_state

ARCHIVE_A_T = RemoteFileMetadata(
    filename="Lalonde_A_T.txt",
    url='http://www.nber.org/~rdehejia/data/nsw_treated.txt',
    checksum=('ab65cd58de17a78b692e66e4d7142192'
              '59ac180428f24c42ddbb928cfb1820fe'))
ARCHIVE_A_C = RemoteFileMetadata(
    filename="Lalonde_A_C.txt",
    url='http://www.nber.org/~rdehejia/data/nsw_control.txt',
    checksum=('8fd745ed2c3426bb77e34b395fb84d45'
              '6d346ba545af58a65b4963160b0699fd'))

ARCHIVE_B_T = RemoteFileMetadata(
    filename="Lalonde_B_T.txt",
    url='http://www.nber.org/~rdehejia/data/nswre74_treated.txt',
    checksum=('e7b742fe0ff07a0f45e129b4ff108bb9'
              '611cd83d53604732c48a8a0a3e20eda3'))
ARCHIVE_B_C = RemoteFileMetadata(
    filename="Lalonde_B_C.txt",
    url='http://www.nber.org/~rdehejia/data/nswre74_control.txt',
    checksum=('a1364cea459d953dc691a667d99194b4'
              'ad335d6d550354fe23a5d2dc58d729b5'))

logger = logging.getLogger(__name__)


def fetch_Lalonde(version="A", data_home=None,
                  download_if_missing=True, random_state=None,
                  shuffle=False, return_X_y=False):
    """Load the Lalonde datasets (uplift regression).

    Download it if necessary.

    There are two versions of the dataset, including 1974 earnings
    (version B) and excluding 1974 earnings (vestion A).
    Source: http://users.nber.org/~rdehejia/data/nswdata2.html

    Parameters
    ----------
    version : string, optional
        Specify which dataset to return.  ``A'' for larger files
        without 1974 earnings, ``B'' for smaller files with 1974
        earnings.

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

    (data, target) : tuple if ``return_X_y`` is True

    """

    if version == "A":
        version_suffix = "_A"
        arch_t = ARCHIVE_A_T
        arch_c = ARCHIVE_A_C
        n_fields = 9
    elif version == "B":
        version_suffix = "_B"
        arch_t = ARCHIVE_B_T
        arch_c = ARCHIVE_B_C
        n_fields = 10
    else:
        raise ValueError("Lalonde dataset version must be A or B")
        
    data_home = get_data_home(data_home=data_home)
    Lalonde_dir = join(data_home, "uplift_sklearn", "Lalonde")
    samples_path = join(Lalonde_dir, "samples" + version_suffix)
    targets_path = join(Lalonde_dir, "targets" + version_suffix)
    treatment_path = join(Lalonde_dir, "treatment" + version_suffix)
    available = exists(samples_path)

    # dictionaries
    feature_names_A = ['age', 'education', 'Black', 'Hispanic',
                       'married', 'nodegree', 'RE75']
    feature_names_B = ['age', 'education', 'Black', 'Hispanic',
                       'married', 'nodegree', 'RE74', 'RE75']
    if version == "A":
        feature_names = feature_names_A
    elif version == "B":
        feature_names = feature_names_B
    else:
        assert False
    target_names = ["RE78"]
    treatment_values = ['not treated', 'treated']

    if download_if_missing and not available:
        if not exists(Lalonde_dir):
            makedirs(Lalonde_dir)
        logger.info("Downloading %s" % arch_t.url)
        archive_path_T = _fetch_remote(arch_t, dirname=Lalonde_dir)
        logger.info("Downloading %s" % arch_c.url)
        archive_path_C = _fetch_remote(arch_c, dirname=Lalonde_dir)
        #print(archive_path)
        #archive_path = "/home/szymon/scikit_learn_data/usklearn_Lalonde/Hillstrom.csv"
        # read the data
        Xy_T = []
        Xy_C = []
        for archive_path in (archive_path_T, archive_path_C):
            is_T = (archive_path == archive_path_T)
            Xy = Xy_T if is_T else Xy_C
            with open(archive_path) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ',
                                       skipinitialspace=True)
                for record in csvreader:
                    record[0] = int(round(float(record[0])))
                    if is_T:
                        assert record[0] == 1
                    else:
                        assert record[0] == 0
                    Xy.append(record)
                    assert len(record) == n_fields, record
            # delete archive
            remove(archive_path)
        Xy = Xy_T + Xy_C
        # decode treatment group
        trt = np.asarray([r[0] for r in Xy], dtype=np.int32)
        # decode targets
        y = np.asarray([float(r[-1]) for r in Xy], dtype=np.float)
        X = [tuple(r[1:-1]) for r in Xy]
        X = np.asarray(X, dtype=np.float)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(y, targets_path, compress=9)
        joblib.dump(trt, treatment_path, compress=9)

    elif not available and not download_if_missing:
        raise IOError("Data not found and `download_if_missing` is False")

    try:
        X, y, trt
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)
        trt = joblib.load(treatment_path)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        trt = trt[ind]
        y = y[ind]

    if return_X_y:
        return X, y, trt

    return Bunch(data=X, target=y, treatment=trt,
                 treatment_values=treatment_values,
                 feature_names=feature_names, target_names=target_names,
                 DESCR=__doc__)
