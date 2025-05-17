**Data Preparation Report**

Below is a high‐level summary of how we transformed the **original CPCSSN “Care4Mind” dataset extraction** into the final prepared data outputs. We detail the **cohort selection criteria**, the **preprocessing pipeline**, and how many records were included or excluded along the way and relationship validation to make sure all the tables only contain the selected 100k sample of patients.

---

## 1. **Source Data Overview**

- **Extracted Data**: We started with multiple CSV files exported from CPCSSN’s “Care4Mind” dataset, which focuses on mental‐health‐related records.  
- **Tables**:  
  - *Patient, PatientDemographic, Encounter, EncounterDiagnosis, FamilyHistory, HealthCondition, Lab, MedicalProcedure, Medication, Referral,* and *RiskFactor.*  
- **Format & Issues**:  
  - Input CSVs were **pipe‐delimited (`|`)** with quoted fields.  
  - The **RiskFactor** table had some **malformed rows** (embedded newlines, extra delimiters). We extracted those anomalies into a separate CSV (`RiskFactor_malformed_rows.csv`) and created a corrected version (`C4MRiskFactor_fixed.csv`) in the extraction folder.  
  - After these fixes, we still faced typical data issues (e.g., out‐of‐range dates, orphaned references) that needed further cleaning.

---

## 2. **Cohort Selection Criteria**

We restricted the dataset to **realistic date ranges** and a **100,000‐patient sample** (stratified by Sex). The key criteria:

1. **Date Range**  
   - **2000‐01‐01** to **2025‐12‐31** for all date columns, such as:
     - *EncounterDate, DateCreated, PerformedDate, StartDate, StopDate*, etc.  
   - Rows with dates **outside** this range were removed from final tables and **collected** in per‐table “outliers” CSV files.

2. **Stratified Sampling by Sex**  
   - After each table was cleaned of out‐of‐range dates, we sampled **100,000** patients from **PatientDemographic** (merged with the “Sex” column in **Patient**), so that *Male/Female/Unknown* categories were as balanced as possible.  
   - (In practice, it ended up slightly under 100k if there weren’t enough rows in certain Sex groups.)

3. **Relationship Integrity**  
   - We excluded references to an `Encounter_ID` that was missing from the **Encounter** table.  
   - For example, we dropped ~230k rows from **EncounterDiagnosis** and ~1.29 million from **Medication** that pointed to an Encounter_ID which no longer existed after filtering.

---

## 3. **Preparation Pipeline**

Below is a **step‐by‐step** outline of the transformations in **`data_preparation_v2.py`**:

1. **Chunked Loading**  
   - Each table was read in chunks of 500,000 rows.  
   - We stripped strings, standardized `_calc` columns to uppercase, and converted date fields to `datetime`.

2. **Out‐of‐Range Dates**  
   - During loading, any row with a date outside **[2000‐01‐01, 2025‐12‐31]** was placed into a separate “outliers” DataFrame for that table.  
   - Finally, each outliers set was saved to files named like **`Encounter_Outside_Date_Range_2000to2025.csv`** within a **`prepared_data/outliers`** folder.

3. **Date‐Validated Tables**  
   - The “main” DataFrame for each table ended up containing only rows that passed the date check.  
   - Example: The **Encounter** table started at ~11.46 million rows but lost ~237k outliers.

4. **Stratified Patient Sampling**  
   - We loaded the **PatientDemographic** table, merged it with **Patient** on `Patient_ID` (to unify the “Sex” field), and performed a **stratified sample** of ~100,000.  
   - The final set of patient IDs is what we call the “cohort.”

5. **Filtering Other Tables**  
   - All other tables (Encounter, Medication, etc.) were restricted to those **Patient_IDs** from the final sample.  
   - Next, we removed any child rows that referenced an Encounter_ID not present in the newly filtered Encounter table.

6. **Relationship Checks & Saves**  
   - We logged warnings if `EncounterDiagnosis` or `Medication` had “orphan” references. Those rows were excluded.  
   - The resulting data was saved to **`*_prepared.csv`** files in the **`prepared_data/`** directory.  
   - For each table, we also computed data quality metrics (row counts, missing percentages, date min/max) and included them in the log.

---

## 4. **Merging Patient & PatientDemographic**

- After the pipeline finished, we performed an additional **merge** of **`Patient_prepared.csv`** with **`PatientDemographic_prepared.csv`** in a notebook (`Patient_&_PatientDemographic_Merge_Notebook.ipynb`).  
- We standardized the “Sex” column so that only **“Male”**, **“Female”**, or **“Unknown”** remain.  
- The final merged file is named **`PatientDemographic_merged_prepared.csv`** with ~100k unique patient records and combined columns from both tables.

---

## 5. **Notable Logs & Figures**

- **Encounter**:  
  - ~237k out‐of‐range rows removed, final ~3.25 million rows remain after filtering to 100k patients.  
- **EncounterDiagnosis**:  
  - ~217k outliers, plus ~230k orphan references removed, final ~3.51 million remain.  
- **Medication**:  
  - ~32k outliers, ~1.29 million orphan references removed, final ~2.20 million remain.  
- **Lab**:  
  - ~2.40 million orphan references removed, final ~2.40 million remain. (Row counts vary widely by table.)

**Sampling**: We ended up with ~99,986 distinct patients (slightly under 100k due to rounding in Sex categories and a small number of `Unknown`).

**Outliers**: Each table’s outliers were logged. For example:
```
Encounter_Outside_Date_Range_2000to2025.csv   -> 237,350 rows
Medication_Outside_Date_Range_2000to2025.csv  -> 32,188 rows
...
```

---

## 6. **Validation**

Finally, to **validate** references:

1. **Check** that each table’s `Patient_ID` is in the final 100k sample.  
2. **Check** that references to `Encounter_ID` in **EncounterDiagnosis**, **Medication**, and **Lab** are found in **Encounter_prepared**.  

We confirmed that after removing out‐of‐range or orphaned rows, all tables are consistent with the 100k patient cohort.

---

## 7. **Conclusion**

By the end of the pipeline:

- **Prepared** versions of each table (`*_prepared.csv`) exist in `prepared_data/`.  
- **Outlier** rows (dates outside 2000–2025) are in `prepared_data/outliers/`.  
- A final **merged** patient/demographic file (`PatientDemographic_merged_prepared.csv`) has ~100k patient records, with standardized “Sex.”  
- All references (Patient_IDs, Encounter_IDs) are consistent across tables within the 100k cohort.

This dataset is now ready for subsequent **analysis** or **research** on the mental health conditions, visits, labs, medications, etc., while maintaining a realistic date range, valid relationships, and a balanced sample of patients by sex.