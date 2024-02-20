"""Utilities for all datasets.

Essentially copied from sklearn.datasets._base to reduce dependency on
unofficial sklearn API."""

from collections import namedtuple, OrderedDict
import hashlib
from urllib.request import urlretrieve
from os.path import join, exists
from os import remove, makedirs
import csv
import logging
from inspect import isfunction

import numpy as np

import joblib
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.datasets import get_data_home

RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _fetch_remote(remote, dirname=None):
    """Helper function to download a remote dataset into path

    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the SHA256 Checksum of the
    downloaded file.

    Parameters
    ----------
    remote : RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum

    dirname : string
        Directory to save the file to.

    Returns
    -------
    file_path: string
        Full path of the created file.
    """

    file_path = (remote.filename if dirname is None
                 else join(dirname, remote.filename))
    urlretrieve(remote.url, file_path)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote.checksum))
    return file_path

def _read_csv(archive_path, feature_attrs, treatment_attrs, target_attrs,
              total_attrs=None, categ_as_strings=False, header=None,
              csv_reader_args={"delimiter":",", "quotechar":'"'}):
    """Read CSV data.

    feature_attrs, treatment_attrs, target_attrs contain descriptions
    of resp. predictive features, treatment description, and targets.
    Currently only a signle treatment attribute is supported.  Each
    description is a list whose elements are tuples describing each
    attribute.  The first element of the tuple is attribute's name,
    the second its type, third element (optional) is the name of
    attribute in the CVS header.  If the type is a sequence, it is
    assumed to be a list of categories.  If the type is a function, it
    will receive as argument a list of columns values and should
    return a transformed list and final numpy dtype.  Otherwise, type
    should be a valid numpy dtype.
    
    If total_attrs is not None, it should contain the total number of
    attributes in each record.

    """
    def parse_attr(Xy_columns, header, attr_description, categ_as_strings):
        """Parse a single attribute.

        Return values and dtype.  Handle categorical attributes correctly."""
        if len(attr_description) == 2:
            attr_name, attr_dtype = attr_description
            file_attr_name = attr_name
        elif len(attr_description) == 3:
            attr_name, attr_dtype, file_attr_name = attr_description
        attr_no = header.index(file_attr_name)
        x = Xy_columns[attr_no]
        if isinstance(attr_dtype, list):
            if categ_as_strings:
                categs = set(attr_dtype)
                for c in x:
                    if c not in categs:
                        raise RuntimeError(f"Unexpected category {c} for attribute {attr_name}")
                maxlen = max(len(c) for c in attr_dtype)
                attr_dtype = f"U{maxlen}"
            else:
                categs = {c:i for i, c in enumerate(attr_dtype)}
                x = [categs[c] for c in x]
                attr_dtype = np.int32
        elif isfunction(attr_dtype):
            x, attr_dtype = attr_dtype(x)
        x = np.array(x, dtype=attr_dtype)
        return x, attr_name

    Xy = []
    with open(archive_path) as csvfile:
        if header is None:
            header = next(csvfile).strip().split(',')
        csvreader = csv.reader(csvfile, **csv_reader_args)
        for record in csvreader:
            Xy.append(record)
            if total_attrs is not None:
                assert len(record) == total_attrs, (record, total_attrs)
    Xy_columns = list(zip(*Xy))
    remove(archive_path)
    
    # parse treatment
    if len(treatment_attrs) != 1:
        raise RuntimeError("Only one treatment is supported")
    trt, attr_name = parse_attr(Xy_columns, header, treatment_attrs[0], categ_as_strings)
    if isinstance(treatment_attrs[0][1], list):
        treatment_values = list(treatment_attrs[0][1])
    else:
        treatment_values = [str(i) for i in range(max(trt) + 1)]

    # parse targets
    targets = OrderedDict()
    for a_descr in target_attrs:
        ta, attr_name = parse_attr(Xy_columns, header, a_descr, categ_as_strings)
        targets[attr_name] = ta
    target_names = list(targets.keys())

    # predictors
    feature_names = []
    columns = []
    for a_descr in feature_attrs:
        fa, attr_name = parse_attr(Xy_columns, header, a_descr, categ_as_strings)
        feature_names.append(attr_name)
        columns.append(fa)
    categ_values = {a[0]:a[1] for a in feature_attrs if isinstance(a[1], list)}
    #X = np.core.records.fromarrays(columns, names=feature_names)

    # create a Bunch
    ret = Bunch(data=columns, treatment=trt,
                feature_names=feature_names,
                target_names=target_names)
    for attr_name in targets:
        ret[attr_name] = targets[attr_name]
    ret.treatment_values=treatment_values
    ret.n_trt=len(treatment_values)-1
    ret.categ_values=categ_values
    return ret

def _fetch_remote_csv(remote, dataset_name,
                      feature_attrs, treatment_attrs, target_attrs,
                      categ_as_strings=False, return_X_y=False,
                      as_frame=False,
                      download_if_missing=True,
                      random_state=None, shuffle=False,
                      header=None, total_attrs=None,
                      csv_reader_args={"delimiter":",", "quotechar":'"'},
                      data_home=None, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__ + "." + dataset_name)
    data_home = get_data_home(data_home=data_home)
    dataset_dir = join(data_home, "uplift_sklearn", dataset_name)
    if categ_as_strings:
        dataset_path = join(dataset_dir, "bunch_str")
    else:
        dataset_path = join(dataset_dir, "bunch")
    available = exists(dataset_path)

    if not available and download_if_missing:
        if not exists(dataset_dir):
            makedirs(dataset_dir)
        logger.info("Downloading %s" % remote.url)
        archive_path = _fetch_remote(remote, dirname=dataset_dir)

        ## read the data
        D =_read_csv(archive_path, feature_attrs=feature_attrs,
                     treatment_attrs=treatment_attrs,
                     target_attrs=target_attrs,
                     total_attrs=total_attrs,
                     csv_reader_args=csv_reader_args,
                     categ_as_strings=categ_as_strings,
                     header=header)

        joblib.dump(D, dataset_path, compress=9)
    elif not available and not download_if_missing:
        raise IOError("Data not found and `download_if_missing` is False")

    # get cached data
    try:
        D
    except NameError:
        D = joblib.load(dataset_path)

    # change columns into a table
    if as_frame:
        import pandas
        D.data = pandas.DataFrame(OrderedDict(zip(D.feature_names, D.data)))
    else:
        all_float = all((x.dtype.kind=="f") for x in D.data)
        if not all_float:
            for i, c in enumerate(D.data):
                D.data[i] = c.astype(object)
        D.data = np.column_stack(D.data)

    if shuffle:
        ind = np.arange(D.data.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        D.data = D.data[ind]
        D.treatment = D.treatment[ind]
        for ta in D.target_names:
            D[ta] = D[ta][ind] 

    if return_X_y:
        X = D.data
        targets = tuple(D[tn] for tn in D.target_names)
        trt = D.trt
        ret = (X,) + targets + (trt,)
    else:
        ret = D
    return ret
