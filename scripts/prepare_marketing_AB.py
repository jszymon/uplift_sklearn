"""Preprocess the marketing_AB dataset from Kaggle."""

import pandas as pd

D = pd.read_csv("/tmp/marketing_AB.csv")

# changes:
# remove record number column ?
# change spaces to _ in column names
# change target from bool to {0,1}

# drop record number column
D = D.drop(columns=D.columns[0])
# rename columns
col_names = list(D.columns)
for i in range(len(col_names)):
    col_names[i] = col_names[i].replace(" ", "_")
D.columns = col_names
# convert target to {0,1}
D.converted = D.converted*1

# print dtypes
print(D.dtypes)

# print dictionaries
for c in D.columns:
    if c not in ["user_id", "total_ads"]:
        print(c, D[c].unique())

D.to_csv("/tmp/marketing_AB.csv_conv", index=False)
