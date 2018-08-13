from ._validation import cross_validate, cross_val_score

from ..metrics import e_sate
from sklearn.metrics import SCORERS
#SCORERS["e_sate"] = make_uplift_scorer(e_sate, greater_is_better=False)
