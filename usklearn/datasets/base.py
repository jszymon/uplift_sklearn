"""Utilities for all datasets.

Essentially copied from sklearn.datasets._base to reduce dependency on
unofficial sklearn API."""

from collections import namedtuple
import hashlib
from urllib.request import urlretrieve
from os.path import join
import csv

import numpy as np

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
              total_attrs=None, categ_as_strings=False, header=None):
    """Read CSV data.

    feature_attrs, treatment_attrs, target_attrs contain descriptions
    of resp. predictive features, treatment description, and targets.
    Currently only a signle treatment attribute is supported.  Each
    description is a list whose elements are tuples describing each
    attribute.  The first element of the tuple is attribute's name,
    the second its type, third elemend (optional) is the name of
    attribute in the header.  If the type is a sequence, it is assumed
    to be a list of categories.  Otherwise, type should be a valid
    numpy dtype.
    
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
        x = np.array(x, dtype=attr_dtype)
        return x, attr_name

    Xy = []
    with open(archive_path) as csvfile:
        if header is None:
            header = next(csvfile).strip().split(',')
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for record in csvreader:
            Xy.append(record)
            if total_attrs is not None:
                assert len(record) == total_attrs, record
    Xy_columns = list(zip(*Xy))

    # parse treatment
    if len(treatment_attrs) != 1:
        raise RuntimeError("Only one treatment is supported")
    trt, attr_name = parse_attr(Xy_columns, header, treatment_attrs[0], categ_as_strings)
    if isinstance(treatment_attrs[0][1], list):
        treatment_values = list(treatment_attrs[0][1])
    else:
        treatment_values = [str(i) for i in range(max(x) + 1)]

    # parse targets
    targets = ()
    target_names = []
    for a_descr in target_attrs:
        ta, attr_name = parse_attr(Xy_columns, header, a_descr, categ_as_strings)
        targets = targets + (ta,)
        target_names.append(attr_name)

    # predictors
    feature_names = []
    columns = []
    for a_descr in feature_attrs:
        fa, attr_name = parse_attr(Xy_columns, header, a_descr, categ_as_strings)
        feature_names.append(attr_name)
        columns.append(fa)
    X = np.core.records.fromarrays(columns, names=feature_names)

    return X, targets, target_names, trt

    #if return_X_y:
    #    ret = (X,) + tuple(targets.values) + (trt,)
    #
    #ret = Bunch(data=X, treatment=trt, n_trt=len(treatment_values),
    #             categ_values=categ_values, treatment_values=treatment_values,
    #             feature_names=feature_names, target_names=target_names,
    #             DESCR=__doc__)
    #for attr_name in target_names:
    #    ret[attr_name] = targets[attr_name]
    #return ret
