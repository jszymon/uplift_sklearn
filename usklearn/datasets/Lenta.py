"""Lenta challenge dataset.

The version used is from the scikit-uplift package with non-ascii
characters removed from the Gender variable.

"""

import numpy as np

from .base import _fetch_remote_csv
from .base import RemoteFileMetadata


ARCHIVE = RemoteFileMetadata(
    filename="lenta_dataset.csv.gz",
    url=('https://github.com/jszymon/uplift_sklearn_data/'
         'releases/download/Lenta/lenta_dataset.csv.gz'),
    checksum=('6fe2d47275d73a3829cc3302cb134c10'
              '93ae464345826f81c6552d39337c8bc7'))


def fetch_Lenta(data_home=None, download_if_missing=True,
                    random_state=None, shuffle=False,
                    categ_as_strings=False, return_X_y=False,
                    as_frame=False):
    """Load the Lenta dataset (uplift classification).

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

    as_frame : boolean, default=False
        If True features are returned as pandas DataFrame.  If False
        features are returned as object or float array.  Float array
        is returned if all features are floats.
    
    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array
        Each row corresponds to the features in the dataset.

    dataset.target_visit : numpy array
        Each value is 1 if website visit occurred 0 otherwise.

    dataset.target_conversion : numpy array
        Each value is 1 if purchase occurred 0 otherwise.

    dataset.target_spend : numpy array
        Each value corresponds to the amount of money spent.

    dataset.DESCR : string
        Description of the Hillstrom dataset.

    (data, target_visit, target_conversion, target_spend) : tuple if
        ``return_X_y`` is True

    """

    # dictionaries
    treatment_values = ['control', 'test']
    gender_values = ["F", "M", "Undefined"]
    
    # attribute descriptions
    treatment_descr = [("group", treatment_values)]
    target_descr = [("target_response_att", np.int32, "response_att")]
]

    feature_descr = [('age', np.int32),
                     ('cheque_count_12m_g20', np.int32),
                     ('cheque_count_12m_g21', np.int32),
                     ('cheque_count_12m_g25', np.int32),
                     ('cheque_count_12m_g32', np.int32),
                     ('cheque_count_12m_g33', np.int32),
                     ('cheque_count_12m_g38', np.int32),
                     ('cheque_count_12m_g39', np.int32),
                     ('cheque_count_12m_g41', np.int32),
                     ('cheque_count_12m_g42', np.int32),
                     ('cheque_count_12m_g45', np.int32),
                     ('cheque_count_12m_g46', np.int32),
                     ('cheque_count_12m_g48', np.int32),
                     ('cheque_count_12m_g52', np.int32),
                     ('cheque_count_12m_g56', np.int32),
                     ('cheque_count_12m_g57', np.int32),
                     ('cheque_count_12m_g58', np.int32),
                     ('cheque_count_12m_g79', np.int32),
                     ('cheque_count_3m_g20',  np.int32),
                     ('cheque_count_3m_g21',  np.int32),
                     ('cheque_count_3m_g25',  np.int32),
                     ('cheque_count_3m_g42',  np.int32),
                     ('cheque_count_3m_g45',  np.int32),
                     ('cheque_count_3m_g52',  np.int32),
                     ('cheque_count_3m_g56',  np.int32),
                     ('cheque_count_3m_g57',  np.int32),
                     ('cheque_count_3m_g79',  np.int32),
                     ('cheque_count_6m_g20',  np.int32),
                     ('cheque_count_6m_g21',  np.int32),
                     ('cheque_count_6m_g25',  np.int32),
                     ('cheque_count_6m_g32',  np.int32),
                     ('cheque_count_6m_g33',  np.int32),
                     ('cheque_count_6m_g38',  np.int32),
                     ('cheque_count_6m_g39',  np.int32),
                     ('cheque_count_6m_g40',  np.int32),
                     ('cheque_count_6m_g41',  np.int32),
                     ('cheque_count_6m_g42',  np.int32),
                     ('cheque_count_6m_g45',  np.int32),
                     ('cheque_count_6m_g46',  np.int32),
                     ('cheque_count_6m_g48',  np.int32),
                     ('cheque_count_6m_g52',  np.int32),
                     ('cheque_count_6m_g56',  np.int32),
                     ('cheque_count_6m_g57',  np.int32),
                     ('cheque_count_6m_g58',  np.int32),
                     ('cheque_count_6m_g79',  np.int32),
                     ('children', np.int32),
                     ('crazy_purchases_cheque_count_12m', np.int32),
                     ('crazy_purchases_cheque_count_1m',  np.int32),
                     ('crazy_purchases_cheque_count_3m',  np.int32),
                     ('crazy_purchases_cheque_count_6m',  np.int32),
                     ('crazy_purchases_goods_count_12m',  np.int32),
                     ('crazy_purchases_goods_count_6m',   np.int32),
                     ('disc_sum_6m_g34', float),
                     ('food_share_15d', float),
                     ('food_share_1m', float),
                     ('gender', gender_values),
                     ('k_var_cheque_15d', float),
                     ('k_var_cheque_3m', float),
                     ('k_var_cheque_category_width_15d', float),
                     ('k_var_cheque_group_width_15d', float),
                     ('k_var_count_per_cheque_15d_g24', float),
                     ('k_var_count_per_cheque_15d_g34', float),
                     ('k_var_count_per_cheque_1m_g24', float),
                     ('k_var_count_per_cheque_1m_g27', float),
                     ('k_var_count_per_cheque_1m_g34', float),
                     ('k_var_count_per_cheque_1m_g44', float),
                     ('k_var_count_per_cheque_1m_g49', float),
                     ('k_var_count_per_cheque_3m_g24', float),
                     ('k_var_count_per_cheque_3m_g27', float),
                     ('k_var_count_per_cheque_3m_g32', float),
                     ('k_var_count_per_cheque_3m_g34', float),
                     ('k_var_count_per_cheque_3m_g41', float),
                     ('k_var_count_per_cheque_3m_g44', float),
                     ('k_var_count_per_cheque_6m_g24', float),
                     ('k_var_count_per_cheque_6m_g27', float),
                     ('k_var_count_per_cheque_6m_g32', float),
                     ('k_var_count_per_cheque_6m_g44', float),
                     ('k_var_days_between_visits_15d', float),
                     ('k_var_days_between_visits_1m', float),
                     ('k_var_days_between_visits_3m', float),
                     ('k_var_disc_per_cheque_15d', float),
                     ('k_var_disc_share_12m_g32', float),
                     ('k_var_disc_share_15d_g24', float),
                     ('k_var_disc_share_15d_g34', float),
                     ('k_var_disc_share_15d_g49', float),
                     ('k_var_disc_share_1m_g24', float),
                     ('k_var_disc_share_1m_g27', float),
                     ('k_var_disc_share_1m_g34', float),
                     ('k_var_disc_share_1m_g40', float),
                     ('k_var_disc_share_1m_g44', float),
                     ('k_var_disc_share_1m_g49', float),
                     ('k_var_disc_share_1m_g54', float),
                     ('k_var_disc_share_3m_g24', float),
                     ('k_var_disc_share_3m_g26', float),
                     ('k_var_disc_share_3m_g27', float),
                     ('k_var_disc_share_3m_g32', float),
                     ('k_var_disc_share_3m_g33', float),
                     ('k_var_disc_share_3m_g34', float),
                     ('k_var_disc_share_3m_g38', float),
                     ('k_var_disc_share_3m_g40', float),
                     ('k_var_disc_share_3m_g41', float),
                     ('k_var_disc_share_3m_g44', float),
                     ('k_var_disc_share_3m_g46', float),
                     ('k_var_disc_share_3m_g48', float),
                     ('k_var_disc_share_3m_g49', float),
                     ('k_var_disc_share_3m_g54', float),
                     ('k_var_disc_share_6m_g24', float),
                     ('k_var_disc_share_6m_g27', float),
                     ('k_var_disc_share_6m_g32', float),
                     ('k_var_disc_share_6m_g34', float),
                     ('k_var_disc_share_6m_g44', float),
                     ('k_var_disc_share_6m_g46', float),
                     ('k_var_disc_share_6m_g49', float),
                     ('k_var_disc_share_6m_g54', float),
                     ('k_var_discount_depth_15d', float),
                     ('k_var_discount_depth_1m', float),
                     ('k_var_sku_per_cheque_15d', float),
                     ('k_var_sku_price_12m_g32', float),
                     ('k_var_sku_price_15d_g34', float),
                     ('k_var_sku_price_15d_g49', float),
                     ('k_var_sku_price_1m_g24', float),
                     ('k_var_sku_price_1m_g26', float),
                     ('k_var_sku_price_1m_g27', float),
                     ('k_var_sku_price_1m_g34', float),
                     ('k_var_sku_price_1m_g40', float),
                     ('k_var_sku_price_1m_g44', float),
                     ('k_var_sku_price_1m_g49', float),
                     ('k_var_sku_price_1m_g54', float),
                     ('k_var_sku_price_3m_g24', float),
                     ('k_var_sku_price_3m_g26', float),
                     ('k_var_sku_price_3m_g27', float),
                     ('k_var_sku_price_3m_g32', float),
                     ('k_var_sku_price_3m_g33', float),
                     ('k_var_sku_price_3m_g34', float),
                     ('k_var_sku_price_3m_g40', float),
                     ('k_var_sku_price_3m_g41', float),
                     ('k_var_sku_price_3m_g44', float),
                     ('k_var_sku_price_3m_g46', float),
                     ('k_var_sku_price_3m_g48', float),
                     ('k_var_sku_price_3m_g49', float),
                     ('k_var_sku_price_3m_g54', float),
                     ('k_var_sku_price_6m_g24', float),
                     ('k_var_sku_price_6m_g26', float),
                     ('k_var_sku_price_6m_g27', float),
                     ('k_var_sku_price_6m_g32', float),
                     ('k_var_sku_price_6m_g41', float),
                     ('k_var_sku_price_6m_g42', float),
                     ('k_var_sku_price_6m_g44', float),
                     ('k_var_sku_price_6m_g48', float),
                     ('k_var_sku_price_6m_g49', float),
                     ('main_format', np.int32),
                     ('mean_discount_depth_15d', float),
                     ('months_from_register', np.int32),
                     ('perdelta_days_between_visits_15_30d', float),
                     ('promo_share_15d', float),
                     ('response_sms', float),
                     ('response_viber', float),
                     ('sale_count_12m_g32', np.int32),
                     ('sale_count_12m_g33', np.int32),
                     ('sale_count_12m_g49', np.int32),
                     ('sale_count_12m_g54', np.int32),
                     ('sale_count_12m_g57', np.int32),
                     ('sale_count_3m_g24', np.int32),
                     ('sale_count_3m_g33', np.int32),
                     ('sale_count_3m_g57', np.int32),
                     ('sale_count_6m_g24', np.int32),
                     ('sale_count_6m_g25', np.int32),
                     ('sale_count_6m_g32', np.int32),
                     ('sale_count_6m_g33', np.int32),
                     ('sale_count_6m_g44', np.int32),
                     ('sale_count_6m_g54', np.int32),
                     ('sale_count_6m_g57', np.int32),
                     ('sale_sum_12m_g24', float),
                     ('sale_sum_12m_g25', float),
                     ('sale_sum_12m_g26', float),
                     ('sale_sum_12m_g27', float),
                     ('sale_sum_12m_g32', float),
                     ('sale_sum_12m_g44', float),
                     ('sale_sum_12m_g54', float),
                     ('sale_sum_3m_g24', float),
                     ('sale_sum_3m_g26', float),
                     ('sale_sum_3m_g32', float),
                     ('sale_sum_3m_g33', float),
                     ('sale_sum_6m_g24', float),
                     ('sale_sum_6m_g25', float),
                     ('sale_sum_6m_g26', float),
                     ('sale_sum_6m_g32', float),
                     ('sale_sum_6m_g33', float),
                     ('sale_sum_6m_g44', float),
                     ('sale_sum_6m_g54', float),
                     ('stdev_days_between_visits_15d', float),
                     ('stdev_discount_depth_15d', float),
                     ('stdev_discount_depth_1m', float),
                     ]


    ret = _fetch_remote_csv(ARCHIVE, "Lenta",
                            feature_attrs=feature_descr,
                            treatment_attrs=treatment_descr,
                            target_attrs=target_descr,
                            categ_as_strings=categ_as_strings,
                            return_X_y=return_X_y, as_frame=as_frame,
                            download_if_missing=download_if_missing,
                            random_state=random_state, shuffle=shuffle,
                            total_attrs=195
                            )
    if not return_X_y:
        ret.descr = __doc__
    return ret
