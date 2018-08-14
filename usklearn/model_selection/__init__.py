from ._validation import cross_validate, cross_val_score
from ..metrics import e_sate, e_satt
from ..metrics import make_uplift_scorer

# register uplift scorers with sklearn
from sklearn.metrics import SCORERS
SCORERS["e_sate"] = make_uplift_scorer(e_sate, greater_is_better=False)
SCORERS["e_satt"] = make_uplift_scorer(e_satt, greater_is_better=False)
