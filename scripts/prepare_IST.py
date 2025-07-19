"""Prepare international stroke trial data."""

import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 400)


os.chdir("/tmp")
if not os.path.isfile("IST_orig.csv"):
    os.system("wget 'https://datashare.ed.ac.uk/bitstream/handle/10283/124/IST_corrected.csv?sequence=5&isAllowed=y' -O IST_orig.csv")

# convert 'ź' character
os.system("tr '\\237' z < IST_orig.csv > IST_conv.csv")

D = pd.read_csv("IST_conv.csv", low_memory=False)

# level H of RXHEP recoded as M for pilot study cases
D.loc[D.RXHEP=="H", "RXHEP"] = "M"

def yn_to_01(x):
    if x.isnull().any():
        mask = x.isnull()
        r = 1.0*(x=="Y")
        r[mask] = pd.NA
        return r
    return 1*(x=="Y")

month_map = {
    'sty': 1,  # Styczeń (January)
    'lut': 2,  # Luty (February)
    'mar': 3,  # Marzec (March)
    'kwi': 4,  # Kwiecień (April)
    'maj': 5,  # Maj (May)
    'cze': 6,  # Czerwiec (June)
    'lip': 7,  # Lipiec (July)
    'sie': 8,  # Sierpień (August)
    'wrz': 9,  # Wrzesień (September)
    'paz': 10, # Październik (October)
    'lis': 11, # Listopad (November)
    'gru': 12  # Grudzień (December)
    }
def convert_date(dte):
    month = dte.str[:3].map(month_map)
    year = dte.str[4:].astype(np.int32)
    return month, year

week_map = {1:"Sun", 2:"Mon", 3:"Tue", 4:"Wed", 5:"Thu", 6:"Fri", 7:"Sat"}
OCCODE_map = {1:"dead", 2:"dependent", 3:"not_recovered", 4:"recovered", 0:"NA", 9:"NA"}


###! for c in D.columns: print(D[c].value_counts(dropna=False))
print(D.pivot_table("ID14", "RXHEP"))
print(D.pivot_table("ID14", "RXASP"))
print(D.pivot_table("ID14", "SET14D"))
print(D.pivot_table("ID14", "ONDRUG"))

res_D = {
    "IS_PILOT": D.RATRIAL.isnull()*1,
    "HOSPNUM": D.HOSPNUM,
    "COUNTRY": D.COUNTRY,
    "RDELAY": D.RDELAY,
    "RCONSC": D.RCONSC,
    "SEX": D.SEX,
    "AGE": D.AGE,
    "RSLEEP": yn_to_01(D.RSLEEP),
    "RATRIAL": yn_to_01(D.RATRIAL),
    "RCT": yn_to_01(D.RCT),
    "RVISINF": yn_to_01(D.RVISINF),
    "RHEP24": yn_to_01(D.RHEP24),
    "RASP3": yn_to_01(D.RASP3),
    "RSBP": D.RSBP,
    "RDEF1": D.RDEF1,    
    "RDEF2": D.RDEF2,    
    "RDEF3": D.RDEF3,    
    "RDEF4": D.RDEF4,    
    "RDEF5": D.RDEF5,    
    "RDEF6": D.RDEF6,    
    "RDEF7": D.RDEF7,    
    "RDEF8": D.RDEF8,    
    "STYPE": D.STYPE,
    "RYEAR": convert_date(D.RDATE)[1],
    "RMONTH": convert_date(D.RDATE)[0],
    "HOURLOCAL": D.HOURLOCAL,
    "MINLOCAL": D.MINLOCAL,
    "DAYLOCAL": D.DAYLOCAL.map(week_map),
    # predictions by external models
    "EXPDD": D.EXPDD,
    "EXPD6": D.EXPD6,
    "EXPD14": D.EXPD14,
    # treatments
    "RXASP": yn_to_01(D.RXASP),
    "RXHEP": D.RXHEP,
    # targets
    "ID14": D.ID14,   # unknown in 2 cases (set to 0) cases identified by the SET14D variable in the original dataset
    "OCCODE": D.OCCODE.map(OCCODE_map),
    # secondary targets/side effects
    "H14": D.H14,
    "ISC14": D.ISC14,
    "NK14": D.NK14,
    "STRK14": D.STRK14,
    "HTI14": D.HTI14,
    "PE14": D.PE14,
    "DVT14": D.DVT14,
    "TRAN14": D.TRAN14,
    "NCB14": D.NCB14,
}

DF_final = pd.DataFrame(res_D)
DF_final.to_csv("/tmp/IST.csv", index=False, na_rep="nan")

