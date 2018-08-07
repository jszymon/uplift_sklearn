"""Hillstrom challenge dataset.

A dataset from Kevin Hillstrom's MineThatData blog. See

    https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

for details.
"""

import logging
from os.path import dirname, exists, join
from os import remove

from sklearn.datasets.base import get_data_home
from sklearn.datasets.base import _fetch_remote
from sklearn.datasets.base import RemoteFileMetadata
from sklearn.utils import Bunch
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
                    random_state=None, shuffle=False, return_X_y=False):
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
    Hillstrom_dir = join(data_home, "usklearn_Hillstrom")
    samples_path = join(Hillstrom_dir, "samples")
    targets_path = join(Hillstrom_dir, "targets")
    available = exists(samples_path)

    if download_if_missing and not available:
        if not exists(Hillstrom_dir):
            makedirs(Hillstrom_dir)
        logger.info("Downloading %s" % ARCHIVE.url)

        archive_path = _fetch_remote(ARCHIVE, dirname=Hillstrom_dir)
        # read the data
        Xy = []
        with open(archive_path) as file_:
            for line in file_.readlines():
                Xy.append(line.replace('\n', '').split(','))
        # delete archive
        remove(archive_path)
        # decode variables
        return Xy
        X = Xy[:, :-1]
        y = Xy[:, -1].astype(np.int32)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(y, targets_path, compress=9)

    elif not available and not download_if_missing:
        raise IOError("Data not found and `download_if_missing` is False")
    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    module_path = dirname(__file__)
    with open(join(module_path, 'descr', 'covtype.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, DESCR=fdescr)
