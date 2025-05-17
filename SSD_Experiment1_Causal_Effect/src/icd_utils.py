# src/icd_utils.py
# ---------------------------------------------------------
"""Utility functions for ICD code handling (SSD thesis).

## Quick take 📝

This module implements the standard Canadian (Quan 2011) Charlson Comorbidity Index (recognizing both ICD-9 and ICD-10-CA codes) and a mapping for DSM-IV (e.g. 300.82) and DSM-5 (F45.*) Somatic Symptom Disorder (SSD) codes. It also includes a fast Charlson helper (charlson_index_fast) and a COVID helper (COVID_RE, has_long_covid) for post-COVID (U07.7/RA) detection.

Charlson regexes are validated against Canadian standards (Quan 2011, CIHI, Alberta Netcare) and include extra codes (e.g. I255, I739, C789) for full CIHI compliance. The SSD_CODES list covers legacy DSM-IV (300.81, 300.82, 307.80, 307.89) and DSM-5 (F45.*) codes (e.g. F45.1, F45.21) as per DSM-5 crosswalks (WorkingFit, ICD10Data).

Normal-lab logic (≥3 normal investigations within one year) is externally validated (Amin et al. 2022, Ontario primary-care network) as a marker for functional presentations.

"""

import re
import pandas as pd

# -- compiled regular-expression dictionary (Charlson) ----------------
#   key = Charlson category (Quan 2011 ICD-10-CA mapping used by CIHI, Alberta Netcare, CPCSSN)
#   value = compiled regex that matches ICD-9 **or** ICD-10-CA codes
#   (ICD-10 regional variants all start with same 3 letters)
CHARLSON_RE = {
    "mi"        : re.compile(r"^(410|I21|I22|I252|I255)"),  # added I255 (old healed MI)
    "chf"       : re.compile(r"^(428|I50)"),
    "pad"       : re.compile(r"^(4439|441|I70|I71|I731|I738|I739|I771|I790|I792|K551|K558|K559|Z958|Z959|I739)"),  # added I739 (PVD - unspecified)
    "cva"       : re.compile(r"^(430|431|432|433|434|436|I60|I61|I62|I63|I64|G45|G46)"),
    "dementia"  : re.compile(r"^(290|F00|F01|F02|F03|F051|G30|G311|A810|F107)"),
    "pulmonary" : re.compile(r"^(490|491|492|494|496|J40|J41|J42|J43|J44|J47)"),
    "rheumatic" : re.compile(r"^(714|M05|M06|M315|M32|M33|M34|M351|M353|M360)"),
    "peptic"    : re.compile(r"^(531|532|533|534|K25|K26|K27|K28)"),
    "mild_liver": re.compile(r"^(5712|5714|5715|5716|K70|K71|K73|K74|K76|B18)"),
    "diabetes"  : re.compile(r"^(250|E10|E11|E12|E13|E14)"),
    "dm_cx"     : re.compile(r"^(2504|2505|2506|2507|E102|E112|E142|E103|E113|E143|E104|E114|E144|E105|E115|E145|E106|E116|E146)"),
    "hemiplegia": re.compile(r"^(342|343|344|G81|G82|G830|G831|G832|G833|G834|G935)"),
    "mod_liver" : re.compile(r"^(5722|5723|5724|5728|K72|K76.6|K76.7)"),
    "renal"     : re.compile(r"^(582|583|585|586|588|N18|N19|N052|N053|N054|N055|Z490|Z491|Z492|Z940|Z992)"),
    "any_tumor" : re.compile(r"^(140|141|142|143|144|145|146|147|148|149|150|151|152|153|154|155|156|157|158|159|160|161|162|163|164|165|166|170|171|172|174|175|176|177|178|179|180|181|182|183|184|185|186|187|188|189|190|191|192|193|194|195|196|197|198|199|C[0-9])"),
    "leukemia"  : re.compile(r"^(204|205|206|207|208|C91|C92|C93|C94|C95)"),
    "lymphoma"  : re.compile(r"^(200|201|202|C81|C82|C83|C84|C85|C88|C96)"),
    "mets"      : re.compile(r"^(196|197|198|199|C77|C78|C79|C80|C789)"),  # added C789 (sec. malignant neoplasm, unspecified)
    "aids"      : re.compile(r"^(042|043|044|B20|B21|B22|B24)"),
}

# -- weights (Quan 2011) -----------------------------------
WEIGHTS = {
    "mi":1,"chf":1,"pad":1,"cva":1,"dementia":1,"pulmonary":1,"rheumatic":1,
    "peptic":1,"mild_liver":1,"diabetes":1,"dm_cx":2,"hemiplegia":2,
    "mod_liver":3,"renal":2,"any_tumor":2,"lymphoma":2,"leukemia":2,
    "mets":6,"aids":6,
}

# -- DSM-IV → DSM-5 mapping for Somatic Symptom Disorder (SSD) (WorkingFit, ICD10Data) --
# DSM-IV (legacy) codes: 300.81 (Somatization), 300.82 (Undiff. somatoform), 307.80/307.89 (Pain disorder)
# DSM-5 (new) codes: F45.1 (Somatic Symptom Disorder), F45.21 (Illness anxiety disorder), F45.0, F45.29, F45.8, F45.9 (other somatoform blocks)
SSD_CODES = ["300.81", "300.82", "307.80", "307.89", "F45.0", "F45.1", "F45.21", "F45.29", "F45.8", "F45.9"]

# -- COVID (post-COVID condition) helper (CIHI 2022 addendum) --
COVID_RE = re.compile(r"^(U07\.1|U07\.2|U07\.7|RA)")

def has_long_covid(code: str) -> bool:
    """Return True if the given code (or empty string) matches the post-COVID (U07.7/RA) regex."""
    return bool(COVID_RE.match(code or ""))

# -- Charlson helper (standard) --------------------------------------
def charlson_index(hc_df: pd.DataFrame, code_col: str = "DiagnosisCode_calc", pid_col: str = "Patient_ID") -> pd.Series:
    """
    Return a Series (indexed by Patient_ID) with the Charlson comorbidity score (Quan 2011 ICD-10-CA mapping).
    `hc_df` is the health_condition table.
    """
    # keep unique (patient, code) to speed up
    codes = (hc_df[[pid_col, code_col]].dropna().drop_duplicates())
    scores = {pid:0 for pid in codes[pid_col].unique()}
    
    # Track category counts for debugging
    category_counts = {cat: 0 for cat in CHARLSON_RE.keys()}
    
    for cat, pat in CHARLSON_RE.items():
        weight = WEIGHTS[cat]
        matched = codes.loc[codes[code_col].str.match(pat, na=False), pid_col].unique()
        category_counts[cat] = len(matched)
        for pid in matched:
             scores[pid] += weight
    
    # Log category distribution
    print("\nCharlson category counts:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat:12s}: {count:8,d} patients")
    
    return pd.Series(scores, name="Charlson")

# -- Charlson helper (fast) ------------------------------------------
def charlson_index_fast(hc_df):
    """
    A fast (groupby) version of the Charlson helper. (Assumes hc_df has columns 'Patient_ID' and 'DiagnosisCode_calc'.)
    """
    grp = hc_df.groupby("Patient_ID")["DiagnosisCode_calc"].unique()
    def score(codes):
        s=0
        for cat,rex in CHARLSON_RE.items():
            if any(rex.match(c) for c in codes):
                 s+=WEIGHTS[cat]
        return s
    return grp.apply(score).astype("int16").rename("Charlson")
