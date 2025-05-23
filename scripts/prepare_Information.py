# script to prepare data from CRAN information package

# conde to run in R first:
# library(Information)
#write.csv(train, file="/tmp/Information_train_R.csv", row.names=F, quote=F)
#write.csv(valid, file="/tmp/Information_valid_R.csv", row.names=F, quote=F)


attr_names = 'TREATMENT,PURCHASE,M_SNC_MST_RCNT_ACT_OPN,TOT_HI_CRDT_CRDT_LMT,RATIO_BAL_TO_HI_CRDT,AGRGT_BAL_ALL_XCLD_MRTG,N_OF_SATISFY_FNC_REV_ACTS,AVG_BAL_ALL_FNC_REV_ACTS,N_BANK_INSTLACTS,M_SNCOLDST_BNKINSTL_ACTOPN,N_FNC_INSTLACTS,N_SATISFY_INSTL_ACTS,M_SNC_MSTREC_INSTL_TRD_OPN,TOT_INSTL_HI_CRDT_CRDT_LMT,M_SNC_OLDST_MRTG_ACT_OPN,M_SNC_MSTRCNT_MRTG_ACT_UPD,M_SNC_MST_RCNT_MRTG_DEAL,N30D_ORWRS_RTNG_MRTG_ACTS,N_OF_MRTG_ACTS_DLINQ_24M,N_SATISFY_PRSNL_FNC_ACTS,RATIO_PRSNL_FNC_BAL2HICRDT,TOT_OTHRFIN_HICRDT_CRDTLMT,N_SATISFY_OIL_NATIONL_ACTS,M_SNCOLDST_OIL_NTN_TRD_OPN,N_BC_ACTS_OPN_IN_12M,N_BC_ACTS_OPN_IN_24M,AVG_BAL_ALL_PRM_BC_ACTS,N_RETAIL_ACTS_OPN_IN_24M,M_SNC_OLDST_RETAIL_ACT_OPN,RATIO_RETAIL_BAL2HI_CRDT,TOT_BAL_ALL_DPT_STORE_ACTS,N_30D_RATINGS,N_120D_RATINGS,N_30D_AND_60D_RATINGS,N_ACTS_WITH_MXD_3_IN_24M,N_ACTS_WITH_MXD_4_IN_24M,PRCNT_OF_ACTS_NEVER_DLQNT,N_ACTS_90D_PLS_LTE_IN_6M,TOT_NOW_LTE,N_DEROG_PUB_RECS,N_INQUIRIES,N_FNC_ACTS_VRFY_IN_12M,N_OPEN_REV_ACTS,N_FNC_ACTS_OPN_IN_12M,HI_RETAIL_CRDT_LMT,N_PUB_REC_ACT_LINE_DEROGS,M_SNC_MST_RCNT_60_DAY_RTNG,N_DISPUTED_ACTS,AUTO_HI_CRDT_2_ACTUAL,MRTG_1_MONTHLY_PAYMENT,MRTG_2_CURRENT_BAL,PREM_BANKCARD_CRED_LMT,STUDENT_HI_CRED_RANGE,AUTO_2_OPEN_DATE_YRS,MAX_MRTG_CLOSE_DATE,UPSCALE_OPEN_DATE_YRS,STUDENT_OPEN_DATE_YRS,FNC_CARD_OPEN_DATE_YRS,AGE,UNIQUE_ID,D_DEPTCARD,D_REGION_A,D_REGION_B,D_REGION_C,D_NA_M_SNC_MST_RCNT_ACT_OPN,D_NA_AVG_BAL_ALL_FNC_REV_ACTS,D_NA_M_SNCOLDST_BNKINSTL_ACTOPN,D_NA_M_SNC_OLDST_MRTG_ACT_OPN,D_NA_M_SNC_MST_RCNT_MRTG_DEAL,D_NA_RATIO_PRSNL_FNC_BAL2HICRDT'
attrs = attr_names.split(",")
attrs = attrs[:2] + sorted(attrs[2:])
attrs.remove("UNIQUE_ID")


for a in attrs:
    print(f"('{a}', float),")
print()
print("total attrs:", len(attrs))


import pandas as pd

def format_f(x):
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s

D = pd.read_csv("/tmp/Information_train_R.csv")
D = D[attrs]
D.to_csv("/tmp/Information_train.csv", index=False, float_format=format_f)

D = pd.read_csv("/tmp/Information_valid_R.csv")
D = D[attrs]
D.to_csv("/tmp/Information_valid.csv", index=False, float_format=format_f)

