---

# **DATA COMPLETENESS AND PRELIMINARY ANALYSIS FOR SOMATIC SYMPTOM DISORDER (SSD) RESEARCH**

**Report Date:** February 28, 2025  
Ryhan Suny  
---

## **1\. IN THIS DOCUMENT,**

I lay out most of my data analytics findings with the goal of checking whether the extracted CPCSSN dataset is “good enough” or rather complete enough to run causal/ML algorithms for at least experiment 1 out of the 5 I planned in a separate document. The jupyter notebook (python) analysis turned out to be 145+ pages in pdf and therefore required me to produce this summary report in writing and in tabular format for better organization. 

The primary objective of this analysis is to evaluate the completeness of the CPCSSN Care4Mind Dataset (February 2025 extraction), align EHR data with DSM-5 criteria for Somatic Symptom Disorder (SSD), and identify refined patient cohorts for in-depth analysis. The aim is to ensure that apparent somatic symptom presentations are not primarily explained by legitimate medical necessity or comorbidity.

### **Highlights and Power:** 

*After refining our cohort to exclude patients with high medical necessity and serious conditions, approximately **7,910 patients (2.7% of the total)** fully meet the combined DSM‐5 criteria for SSD—indicating a robust sample for our causal mediation study. This subgroup is derived from 56,035 patients with multi-system somatic symptoms (15.91%), 44,208 with persistent anxiety (12.55%), and 49,081 with excessive utilization (13.94%), providing **sufficient power** for the analysis. While structural equation modeling literature occasionally recommends 3,000–5,000 participants for complex models, Fritz and MacKinnon (2007) demonstrate that for single-mediator designs, a sample size of 462 achieves 80% power to detect small effects (standardized path coefficients around 0.14). Our sample size far exceeds this threshold, even under conservative assumptions.* 

---

## **2\. DATA COMPLETENESS CHECKS**

### **2.1 Lab Results Frequency Analysis**

| Metric | Count | Percentage | Notes / Justification |
| ----- | ----- | ----- | ----- |
| Total lab records | 8,528,807 | 100.00% | Direct count from **Lab\_prepared.csv**. |
| Numeric **TestResult\_calc** | 4,304,804 | 50.47% | Converted using `pd.to_numeric()`. |
| Non-numeric or missing **TestResult\_calc** | 4,224,003 | 49.53% | Non-convertible or missing entries. |
| Complete normal range data (Upper & Lower Normal) | 1,219,701 | 14.30% | Only rows with both `UpperNormal` and `LowerNormal` populated. |
| Top lab tests in **Name\_calc** | – | – | TOTAL CHOLESTEROL, HDL, LDL, TRIGLYCERIDES, FASTING GLUCOSE. |

About **14%** of lab rows have upper/lower normal limits, enabling automatic “normal vs. abnormal” classification. A personalized **12-month window** was defined per patient (index date \= most recent lab), within which **37.60%** of those with valid range data had ≥3 normal labs, suggesting “excessive” testing for some.

---

### **2.2 Anxiety-Related Prescription Analysis**

| Metric | Value | Notes / Justification |
| ----- | ----- | ----- |
| Medication table dimensions | 7,706,628 rows, 36 columns | Loaded from **Medication\_prepared.csv**. |
| Missing `Name_calc` entries | 315,680 | 4.10% of all rows lack a medication name. |
| Total anxiety medications prescribed | 949,107 | 12.3% of all prescriptions (SSRIs, SNRIs, benzodiazepines, other anxiolytics, etc.). |
| Patients prescribed anxiety meds | 122,486 of 292,050 | 41.9% of those with any medication record. |
| Persistent anxiety medication use (≥6 mo) | 77,896 | 26.7% of total medication patients. |
| Patients meeting DSM-5 B2 and B3\* | 7,910 | 2.7% of the total population (persistent anxiety \+ excessive test usage). |

“Persistent health anxiety” for Criterion B2 is operationalized as ≥6 months of continuous medication.

---

### **2.3 Somatic Symptom Patterns Analysis (Criterion A)**

| Metric | Count | Percentage |
| ----- | ----- | ----- |
| Encounter diagnosis records | 12,471,764 | 100.00% |
| Somatic symptom diagnoses | 789,181 | 6.33% of all diagnoses |
| Patients with ≥1 somatic symptom diagnosis | 197,154 | 55.98% of 352,161 |
| Multiple body systems (≥2 somatic systems) | 93,176 | 26.46% of 352,161 |
| Patients after refinement\*\* | 56,035 | 15.91% of 352,161 |

After excluding cases with high medical necessity (e.g., advanced comorbidity), \~15.91% remain as potential Criterion A.

---

### **2.4 Healthcare Utilization & Doctor Shopping Analysis (Criterion B3)**

| Utilization Measure | Count | Percentage | Definition / Threshold |
| ----- | ----- | ----- | ----- |
| Total encounters | 11,577,739 | 100.00% | All rows in **Encounter\_prepared.csv**. |
| High utilizers (≥95th percentile of total) | 17,612 | 5.00% | ≥100 total encounters. |
| Average ≥2 visits/month | 51,676 | 14.68% | Calculated from monthly grouping in prior 12 months. |
| Doctor shoppers (≥90th percentile, ≥5 providers) | 47,646 | 13.54% | Based on distinct `Provider_ID`. |
| ≥2 B3 indicators | 75,362 | 22.41% | E.g., doctor shopping \+ high visits, or ED usage \+ repeated complaint. |
| Strict B3 (≥3 indicators) | 8,877 | 2.52% | Must have three different B3 flags simultaneously. |

A **12-month anchor** around each patient’s final encounter date was used to count encounters and providers. About 2.52% reached three distinct B3 indicators.

---

### **2.5 Comorbidity & Medical Necessity Analysis**

| Comorbidity | Count | Percentage |
| ----- | ----- | ----- |
| Hypertension | 69,301 | 19.68% |
| Cancer | 53,180 | 14.97% |
| Asthma/COPD | 43,330 | 12.31% |
| Multiple chronic conditions (≥2) | 76,072 | 21.60% |
| High Charlson Index (≥3) | 14,415 | 4.09% |

A **Charlson Comorbidity Index** was calculated based on ICD‑9/ICD‑10 mappings. An additional **Medical Necessity Score** was computed by combining the Charlson Index with other chronic conditions, using weighted sums. A simple linear regression model predicted expected encounters: 

## y=0+1M 

where MMM is the Medical Necessity Score. The **Utilization Ratio** is defined as:

## Utilization Ratio=Actual Encounters​y+1

Patients above the 90th percentile of the ratio (approximately 2.02) were flagged as “excessive users.” **High ratio patients:** 35,613 (10.12%).

---

## **3\. DSM-5 ALIGNMENT: REFINEMENT & RESULTS**

| Criterion | Before Refinement | After Refinement | Rationale |
| ----- | ----- | ----- | ----- |
| **A: Somatic symptoms** (≥2 systems) | 93,176 (26.46%) | 56,035 (15.91%) | Excluded serious conditions (cancer, Charlson ≥3). |
| **B2: Persistent anxiety ≥6 mo** | 77,896 (22.12%) | 44,208 (12.55%) | Removed those with legitimate medical reasons for anxiety. |
| **B3: Excessive utilization** | 75,362 (21.40%) | 49,081 (13.94%) | Excluded medically justified high usage (normal ratio). |

Refined cohorts are notably smaller once legitimate comorbidity-driven utilization is excluded.

---

## **4\. POSSIBLE NEXT STEPS**

| Possible Steps | Description |
| ----- | ----- |
| **Establish domain-specific lab cutoffs** | Recapture more than 14% of labs using recognized cutpoints for HbA1c, TSH, etc. |
| **Use derived metrics in causal mediation** | Investigate how “excessive lab testing” mediates the link between health anxiety and health outcomes. |
| **Consult clinical experts** | Confirm that “excessive” labs or provider visits are genuinely unwarranted and refine ≥5 providers threshold as needed. |
| **Explore nonlinear or Advanced ML models** | A more flexible approach to predicting expected encounters may better capture real-world utilization than a simple linear regression. |

---

## **5\. METHODOLOGICAL SUMMARY**

| Step | Method |
| ----- | ----- |
| Data Completeness Checks | Assessed missingness in numeric columns and coverage of normal ranges using pandas. |
| Criterion A (Somatic Symptoms) | Grouped ICD codes into body systems; defined persistence as ≥180 days between first and last encounter. |
| Criterion B2 (Persistent Anxiety) | Identified anxiety medications (SSRIs, SNRIs, benzodiazepines, etc.) and measured duration (≥6 months). |
| Criterion B3 (Excessive Utilization) | Calculated encounter percentiles and flagged high utilizers, doctor shoppers, and repeated visits. |
| Comorbidity & Medical Necessity | Computed the Charlson Comorbidity Index; built a Medical Necessity Score and derived a Utilization Ratio. |
| Regression Modeling | Fitted a simple linear regression to predict expected encounters from the Medical Necessity Score. |
| 12‑Month Windows | Applied a retrospective 12‑month window for both labs and encounters using each patient’s most recent date. |

---

## **6\. SUMMARY**

Data completeness strongly affects SSD research reliability. Only \~14% of lab rows readily provide normal/abnormal flags, and a significant fraction of high encounter counts appear justified by serious conditions. By excluding medical-necessity-driven utilizers, we narrow potential SSD cohorts. This refined alignment with DSM-5 criteria forms a solid foundation for further causal modeling, prospective validation, and eventual clinical integration.

---

# **APPENDIX: DETAILED METHODOLOGICAL EXPLANATIONS**

The following sections provide **more thorough detail** on key analytical steps and short code snippets that were part of the original notebook.

## **A1. Lab Classification: Normal vs. Abnormal**

1. **Parsing Normal Ranges**  
   * We used columns `LowerNormal` and `UpperNormal` when **both** were available (\~14% of lab rows).  
   * A row’s **is\_normal** was set to `True` if   
     LowerNormal≤TestResult\_calc≤UpperNormal

2. **Personalized 12-Month Window**  
   * For each patient, define an **index date \= most recent lab date**.  
   * Include only labs within 365 days **before** that index date.  
   * Count how many are flagged **is\_normal**.  
   * “Excessive testing” typically required **≥3** normal labs in that personalized window.

## **A2. Anxiety Medication: Persistent Use & B2**

1. **Defining Anxiety Med Classes**  
   * SSRIs (e.g., SERTRALINE, ESCITALOPRAM)  
   * SNRIs (VENLAFAXINE, DULOXETINE)  
   * Benzodiazepines (DIAZEPAM, LORAZEPAM), etc.  
2. **Medication Duration**  
   * We used columns like `StartDate` and `StopDate` to compute “days on medication.” duration\_days≥180 days  
   * **Persistent** anxiety medication use \= **≥180 days**.  
3. **B2 Intersection**  
   * This set of patients was matched with those having ≥3 normal labs (excessive tests), forming a subset that meets both **B2** and **B3**.  
     

## **A3. Somatic Symptom Detection (Criterion A)**

1. **ICD-Based Body Systems**  
   * For each row in `EncounterDiagnosis`, we matched `DiagnosisCode_calc` to regular expressions for general, GI, GU, musculoskeletal, etc.  
   * A new column `is_somatic_symptom` flagged if any body-system symptom was found.  
2. **Temporal Persistence**  
   * For each patient’s somatic-symptom-coded encounters, we checked the earliest vs. latest **EncounterDate**.  
   * If latest−earliest ≥ 180 days, the patient had “persistent” symptoms.

## **A4. Doctor Shopping & B3 Indicators**

1. **Encounter Window**  
   * Similar to labs, we anchored each patient to the final `EncounterDate` and looked at prior 12 months.  
2. **High Utilization**  
   * We computed each patient’s total encounters in that window.  
   * 90th or 95th percentile cutoffs flagged “high utilizers.”  
3. **Provider Diversity** (“Doctor Shopping”)  
   * `provider_count` \= the number of distinct `Provider_ID` per patient.  
   * 90th percentile (≥5 providers) was used as the threshold.  
4. **Strict B3**  
   * We enumerated how many indicators each patient had (e.g., *doctor\_shopping=1, repeated\_symptom\_visits=1, frequent\_ED\_use=1*, etc.)  
   * B3 strict required ≥3 different indicators.

## **A5. Comorbidity: Charlson Index & Medical Necessity**

1. **Charlson Mapping**  
   * ICD-9/10 patterns (e.g., `^250` for diabetes, `^I50` for heart failure) were used.  
   * Each patient got a “charlson\_index” equal to the sum of relevant condition weights.  
2. **Medical Necessity Score**  
   * Additional chronic conditions (e.g., cancer, autoimmune) contributed partial weights.  
   * The sum gave a “medical\_necessity\_score.”Medical Necessity Score=i=1niCi    
     where Ci is an indicator for the *i’*th condition and i its weight.  
3. **Linear Model for Expected Encounters**

   * Expected encounters were predicted using:

## y=0+1M 

* We fit `encounter_count` vs. `medical_necessity_score` in a simple linear regression:

  *model \= LinearRegression()*

  *X \= combined\_df\[\['medical\_necessity\_score'\]\].values*

  *y \= combined\_df\['encounter\_count'\].values*

  *model.fit(X, y)*

  *combined\_df\['expected\_encounters'\] \= model.predict(X)*

4. **Utilization Ratio**   
- The Utilization Ratio was then computed as:

## Utilization Ratio=Actual Encounters​y+1

Patients above the 90th percentile (\~2.02) are flagged as “excessive beyond comorbid justification.”

## **A6. Refined Exclusions & Final SSD Cohort**

1. **Exclude Serious Medical Conditions**  
   * Cancer, advanced heart disease, high Charlson (≥3).  
   * Anxiety explained by major illnesses was also removed.  
2. **Exclude Those with Normal Utilization Ratio**  
   * We wanted only those whose usage was “excessive” after controlling for comorbidity.  
3. **Intersection of A \+ B2 \+ B3**  
   * Recompute final subsets with medical necessity removed.  
   * This refined approach drastically shrinks the cohorts.  
4. **Unexplained Symptoms**  
   * In some expansions, we also tested how many ICD-coded symptoms had no corresponding condition to explain them (e.g., GI symptoms in someone with no GI disease).  
   * “Unexplained” or “symptom-condition mismatch” could further refine Tier 1 or Tier 2 SSD.

## **A7. Example Code Snippet for Intersection of Criteria**

*criterion\_a\_patients \= set(... )  \# A: persistent multi-system*  
*criterion\_b2\_patients \= set(... ) \# B2: persistent anxiety*  
*criterion\_b3\_patients \= set(... ) \# B3: excessive usage*  
*full\_dsm5\_pattern \= criterion\_a\_patients.intersection(criterion\_b2\_patients).intersection(criterion\_b3\_patients)*

Then we removed any with `has_serious_condition == 1` or `high_utilization_ratio == 0` if that disqualified them.

---

**Note:** This Appendix summarizes additional logic and short code segments that guided the thorough analysis. Actual scripts, including data-loading commands and intermediate data merges, appear in the original Jupyter notebook.

# References:

Fritz, M. S., & MacKinnon, D. P. (2007). Required sample size to detect the mediated effect. *Psychological Science, 18*(3), 233–239.  
Preacher, K. J., & Kelley, K. (2011). Effect size measures for mediation models: Quantitative strategies for communicating indirect effects. *Psychological Methods, 16*(2), 93–115.

NEXT STEPS AS OF MARCH 10

1. **ICD-9 Codes for "Not Yet Diagnosed" (NYD)** \- Identifying specific codes or synonyms that indicate a provisional or undetermined diagnosis in ICD-9.  
2. **Referral Order Analysis** \- Investigating how referral orders (psychiatry vs. other specialists) are structured in real-world datasets, using academic and literature-backed methodologies.  
3. **Conceptual Framework for Patient-Level Table Construction** \- Outlining a structured approach in Python to merge relevant tables (Encounters, Lab, Referral, etc.) to create the required dataset while ensuring proper justification and references.

### **1\. ICD-9 Codes for "Not Yet Diagnosed" (NYD)**

**ICD-9 Coding for Undetermined Diagnosis:** ICD-9-CM does not have a specific code literally named "Not Yet Diagnosed," but it provides codes for situations where a definitive diagnosis isn’t established. A key example is **ICD-9 code 799.9**, defined as “Other unknown and unspecified cause of morbidity and mortality,” which is commonly used to indicate an undetermined or deferred diagnosis ([Vital and Health Statistics; Series 2, No. 104 (7/87)](https://www.cdc.gov/nchs/data/series/sr_02/sr02_104.pdf#:~:text=of%20diagnosis%20deferred%20or%20for,on%20special%20conditions%20and%20examinations)). In fact, coding guidelines (e.g. for the National Ambulatory Medical Care Survey) instructed that when a physician noted *“diagnosis deferred”*, the encounter should be coded with 799.9 ([Vital and Health Statistics; Series 2, No. 104 (7/87)](https://www.cdc.gov/nchs/data/series/sr_02/sr02_104.pdf#:~:text=of%20diagnosis%20deferred%20or%20for,on%20special%20conditions%20and%20examinations)). This essentially flags that the patient’s condition is *provisional/unknown* at that visit. Similarly, in psychiatric settings using DSM-IV multiaxial diagnosis, the code **V71.09** could be used to mean “No diagnosis on Axis I or II” (i.e. no definitive mental health diagnosis was made) ( ). In practice, clinicians might label a problem as *“NYD”* or *“Diagnosis deferred”* in notes, and coders would translate that to these ICD-9 codes or to symptom codes.

**“Rule-out” and Provisional Diagnoses:** Common terminology for provisional diagnoses includes phrases like *“probable,” “suspected,” “questionable,” “rule out (R/O),”* or *“working diagnosis.”* However, official ICD-9 coding rules advise **against using uncertain diagnoses codes in outpatient data**. Instead, the coder should record the **signs/symptoms** or most certain finding if a definite diagnosis isn’t confirmed ([Proper Diagnosis Under ICD-10-CM: Don’t be a Coding ‘Minimalist’](https://www.hmpgloballearningnetwork.com/site/twc/articles/proper-diagnosis-under-icd-10-cm-dont-be-coding-minimalist#:~:text=first,Symptoms%2C%20Signs%2C%20and%20Abnormal%20Clinical)). For example, if a patient presents with chest pain NYD, the encounter might be coded as chest pain (symptom) rather than an assumed diagnosis. This rule is emphasized in ICD-9 guidelines: terms such as “probable” or “rule out” indicate uncertainty and **should not be coded as if they were established** in outpatient settings ([Proper Diagnosis Under ICD-10-CM: Don’t be a Coding ‘Minimalist’](https://www.hmpgloballearningnetwork.com/site/twc/articles/proper-diagnosis-under-icd-10-cm-dont-be-coding-minimalist#:~:text=first,Symptoms%2C%20Signs%2C%20and%20Abnormal%20Clinical)). (In inpatient coding, tentative diagnoses at discharge *could* be coded as if confirmed, but this does not apply to outpatient visits ([C:\\MyFiles\\9cmguidelines02e3.wpd](https://med.jax.ufl.edu/compliance/documents/icd_guidelines.pdf#:~:text=diagnosis%20applies%20only%20to%20inpatients,used%20in%20lieu%20of%20principal)).) Thus, *NYD cases in real-world datasets often appear as either symptom codes or generic “unknown” codes* rather than a specific disease code.

**Real-World Coding Patterns:** In practice, when a diagnosis is pending or unclear, providers might temporarily use a nonspecific code. Aside from 799.9, ICD-9’s “Symptoms, Signs, and Ill-defined Conditions” chapter (780–799) contains many codes that get used when the exact cause is not yet diagnosed ([Proper Diagnosis Under ICD-10-CM: Don’t be a Coding ‘Minimalist’](https://www.hmpgloballearningnetwork.com/site/twc/articles/proper-diagnosis-under-icd-10-cm-dont-be-coding-minimalist#:~:text=first,Symptoms%2C%20Signs%2C%20and%20Abnormal%20Clinical)). For instance, a patient being evaluated for various complaints without a conclusion might carry diagnoses like “780.6 Fever, unspecified” or “786.50 Chest pain, unspecified” in the interim. Another strategy is using **“observation” codes (V71.x)** for cases where a serious condition was suspected but *ultimately not found*. For example, code V71.4 is *“Observation for suspected cardiovascular condition not found”*. These V71 codes communicate that the patient was examined for a possible illness which was *ruled out*. In summary, **ICD-9 provides ways to denote an undiagnosed status** – either through explicit codes for no definitive diagnosis (e.g. 799.9, V71.09) or by coding the presenting complaint – and **coding guidelines and literature reinforce using these rather than prematurely assigning a specific disease code** ([Proper Diagnosis Under ICD-10-CM: Don’t be a Coding ‘Minimalist’](https://www.hmpgloballearningnetwork.com/site/twc/articles/proper-diagnosis-under-icd-10-cm-dont-be-coding-minimalist#:~:text=first,Symptoms%2C%20Signs%2C%20and%20Abnormal%20Clinical)) ([Vital and Health Statistics; Series 2, No. 104 (7/87)](https://www.cdc.gov/nchs/data/series/sr_02/sr02_104.pdf#:~:text=of%20diagnosis%20deferred%20or%20for,on%20special%20conditions%20and%20examinations)). This approach is echoed in academic discussions of “diagnosis deferred,” ensuring that data reflect the uncertainty (for example, one analysis of ambulatory care data found 799.9 was frequently used when the provider had not yet established a diagnosis) ([Vital and Health Statistics; Series 2, No. 104 (7/87)](https://www.cdc.gov/nchs/data/series/sr_02/sr02_104.pdf#:~:text=of%20diagnosis%20deferred%20or%20for,on%20special%20conditions%20and%20examinations)). Such codes or notations serve as synonyms for “NYD” in datasets, flagging patients whose condition is still under investigation.

### **2\. Referral Order Analysis (Psychiatry vs. Other Specialists)**

**Structure of Referral Entries in EHR:** Electronic Health Records typically record referrals as distinct order entries or records, capturing details like the **referral date**, the **specialty or target provider**, and sometimes the reason. For example, when a primary care provider refers a patient, they enter a referral order specifying the specialist (e.g. Neurology, Orthopedics, Psychiatry). In a research database or EHR extract, referrals often appear in a dedicated table (or as part of an “orders” table) with fields for patient ID, referring provider, referred-to **specialty**, and the date of the referral. Best practices from health IT standards (like HL7 FHIR) also model referrals as separate resources linked to the patient, indicating the requested service (specialty) and date. In summary, **referral data is structured as time-stamped events pointing to a specialty**, which allows us to reconstruct the sequence of consultations for a patient ( [Identifying new referrals from FPs using EMRs \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4196822/#:~:text=The%20CPCSSN%202012%20data%20contained,outgoing%E2%80%94referrals%20from%20incoming%20consultation%20reports) ).

**Extracting Referral Sequences:** To analyze the order of referrals (e.g. whether a patient saw other specialists before Psychiatry), we need to sort each patient’s referrals chronologically. A straightforward methodology is: **group referrals by patient and sort by referral date**. Once sorted, one can identify the first referral, second referral, and so on for that patient. In our case, we specifically want to compare *when psychiatry was consulted relative to other specialties*. A practical approach is to split the referrals into two categories – **psych vs. non-psych** – then compute metrics like “date of first psychiatry referral” and “date of first other-specialist referral” for each patient. By comparing these dates, we can determine the order (e.g. *Psychiatry was referred after other specialists* if the first psych referral date comes later).

**Literature and Best Practices:** Research on referral patterns often emphasizes cleaning and classifying referral data properly before sequence analysis. For instance, Ryan et al. (2013) had to *distinguish referrals to specialist physicians from those to allied health* and identify outgoing referral orders (versus incoming consult letters) in a primary care EMR dataset ( [Identifying new referrals from FPs using EMRs \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4196822/#:~:text=The%20CPCSSN%202012%20data%20contained,outgoing%E2%80%94referrals%20from%20incoming%20consultation%20reports) ). Their methodology involved filtering referral records by type of specialist and using the timestamps to trace referral sequences. This example underlines two best practices for our scenario: (1) **ensure only the intended referral types are included** (in our case, we’d include referrals to medical specialists and to psychiatrists, and exclude, say, referrals to physical therapy or other services if those are present), and (2) **use the referral initiation date** as the key for ordering events. Another best practice is to consider the context: often the primary care physician is the source of referrals, so the sequence essentially reflects the PCP’s decision order. Academic studies of referral flows have used **sequence analysis** and simple chronologic sorting to understand care pathways. For example, a Canadian study describing referral patterns sorted referrals by date to see how patients move from family doctors to specialists ( [Identifying new referrals from FPs using EMRs \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4196822/#:~:text=The%20CPCSSN%202012%20data%20contained,outgoing%E2%80%94referrals%20from%20incoming%20consultation%20reports) ). In our analysis, after sorting by date we might create indicators such as *“Saw Psychiatry first”* vs *“Saw other specialist first.”* If multiple referrals occur on the same day, one might need to examine time or assume no clear priority unless domain knowledge says otherwise.

**Justifying the Methodology:** Determining referral order by date is a logical choice because referrals are inherently time-ordered events in a patient’s journey. By reviewing the sequence, we can infer potential care patterns (e.g., a common pattern for somatic symptom cases might be **referral to several medical specialists first, and only later referral to Psychiatry** when medical workups are unrevealing). This approach is supported by real-world examples: patients with unexplained symptoms often undergo extensive specialty consultations before mental health referral ([Somatic Symptom Disorder \- StatPearls \- NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK532253/#:~:text=Limited%20laboratory%20testing%20is%20recommended,Excessive)). Clinical guidance on somatic symptom disorder, for instance, notes that patients usually get a thorough medical workup and even multiple specialty evaluations, and *only after ruling out medical causes* do they engage with mental health services ([Somatic Symptom Disorder \- StatPearls \- NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK532253/#:~:text=Limited%20laboratory%20testing%20is%20recommended,Excessive)). By extracting the referral sequence, we can test such patterns against the data. Ensuring the integrity of this sequence extraction (through proper data cleaning, using all relevant referral records, and validating specialty categories) aligns with best practices in health informatics research, where accurate temporal ordering is crucial for trajectory analysis ( [Identifying new referrals from FPs using EMRs \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4196822/#:~:text=The%20CPCSSN%202012%20data%20contained,outgoing%E2%80%94referrals%20from%20incoming%20consultation%20reports) ). In summary, **sorting referrals chronologically and categorizing by specialty (psychiatry vs others) is a justified and literature-backed method** to analyze referral order, enabling us to say, for example, “X% of patients were referred to psychiatry only after seeing another specialist first,” with confidence in the data sequence.

### **3\. Conceptual Framework for Patient-Level Table Construction**

To create a patient-level dataset from multiple tables (e.g. *EncounterDiagnosis*, *Lab*, *Referral*), we need to **merge and aggregate data by patient**. This involves linking records using patient identifiers, aligning time frames where relevant, and engineering new features/columns that summarize the patient’s journey. Below is a conceptual outline of the logic, with justification at each step:

**Step 1: Identify “NYD” Status via Diagnoses.** Start with the EncounterDiagnosis table to flag cases of “Not Yet Diagnosed.” We define a set of ICD-9 codes that imply an undetermined diagnosis – for example, `799.9` (unspecified morbidity) and `V71.xx` observation codes, as discussed earlier ([Vital and Health Statistics; Series 2, No. 104 (7/87)](https://www.cdc.gov/nchs/data/series/sr_02/sr02_104.pdf#:~:text=of%20diagnosis%20deferred%20or%20for,on%20special%20conditions%20and%20examinations)). We scan each patient’s encounter diagnoses for any of these codes or even textual labels like “NYD.” If any encounter for a patient carries an NYD-indicative code, we mark that patient with a flag `NYD_status = True`. This step ensures we capture patients who at some point had no clear diagnosis. The choice of codes is grounded in ICD-9 documentation (ensuring we use official codes for “no diagnosis”) and in real-world coding practice (e.g., use of 799.9 for deferred diagnosis) ([Vital and Health Statistics; Series 2, No. 104 (7/87)](https://www.cdc.gov/nchs/data/series/sr_02/sr02_104.pdf#:~:text=of%20diagnosis%20deferred%20or%20for,on%20special%20conditions%20and%20examinations)). By using standardized code lists, we adhere to best practices of leveraging controlled vocabularies in healthcare data analysis (which improves reproducibility and correctness of our cohort selection).

*Justification:* Identifying NYD patients is crucial because they form the subset of interest (those whose condition wasn’t immediately diagnosed). Using ICD codes for this is reliable since coders were instructed to use specific codes for “diagnosis deferred” scenarios ([Vital and Health Statistics; Series 2, No. 104 (7/87)](https://www.cdc.gov/nchs/data/series/sr_02/sr02_104.pdf#:~:text=of%20diagnosis%20deferred%20or%20for,on%20special%20conditions%20and%20examinations)). This strategy is supported by the notion that “diagnoses often are not established at the initial visit” and one must rely on interim codes ([Proper Diagnosis Under ICD-10-CM: Don’t be a Coding ‘Minimalist’](https://www.hmpgloballearningnetwork.com/site/twc/articles/proper-diagnosis-under-icd-10-cm-dont-be-coding-minimalist#:~:text=first,Symptoms%2C%20Signs%2C%20and%20Abnormal%20Clinical)). Our patient-level table will thus have a column like `NYD_flag` indicating whether the patient experienced an undiagnosed period.

**Step 2: Integrate Laboratory Results (Normal vs Abnormal).** Next, incorporate data from the Lab table to understand if the patient’s workup showed any abnormalities. We link lab results to patients (using Patient ID; if lab records also have encounter IDs, we can optionally align them with the specific NYD encounter). For each patient, we determine whether lab tests were *normal* or *abnormal*. Many clinical lab systems provide an **abnormal flag** on each result – for example, a field that might contain “H” (high), “L” (low), or “N” (normal) ([Flag Abnormal Lab Results: Why & Difficulties. \- Gregory Schmidt](http://www.gregoryschmidt.ca/writing/flag-abnormal-lab-results-why#:~:text=Schmidt%20www,N)) ([MIMIC Observation Labevents \- Definitions \- MIMIC Implementation Guide v1.3.0](https://mimic.mit.edu/fhir/StructureDefinition-mimic-observation-labevents-definitions.html#:~:text=Historically%20used%20for%20laboratory%20results,flow%20sheets%20to%20signal%20the)). We can leverage such a field to categorize each lab result. A simple algorithm: for each patient, check all relevant lab results in the period of interest (e.g., around the NYD encounter); if **none** of the results are abnormal (i.e., all are within normal range), set `All_labs_normal = True` for that patient, otherwise False. This condenses potentially many lab entries into a single patient-level indicator. In Python/pandas, this could be done by grouping the Lab dataframe by patient and evaluating a condition on the “abnormal flag” column (if available). If an abnormal-flag isn’t directly given, we could compare numeric results to reference ranges – but typically an EHR lab table includes either a flag or reference range fields ([MIMIC Observation Labevents \- Definitions \- MIMIC Implementation Guide v1.3.0](https://mimic.mit.edu/fhir/StructureDefinition-mimic-observation-labevents-definitions.html#:~:text=Historically%20used%20for%20laboratory%20results,flow%20sheets%20to%20signal%20the)).

*Justification:* This step is justified because **normal test results are a key part of identifying somatic symptom patterns**. Patients with psychosomatic or unexplained symptoms often have a litany of normal investigations ([Somatic Symptom and Related Disorders (SSRDs) | Children's Hospital of Philadelphia](https://www.chop.edu/conditions-diseases/somatic-symptom-and-related-disorders-ssrds#:~:text=It%20is%20important%20to%20note,time%20to%20start%20appropriate%20treatment)). By capturing this in our dataset (e.g., a patient who had all labs normal despite complaints), we create a feature that, according to clinical literature, correlates with the likelihood of a somatic symptom disorder or other functional diagnosis (since organic causes were not evidenced) ([Somatic Symptom and Related Disorders (SSRDs) | Children's Hospital of Philadelphia](https://www.chop.edu/conditions-diseases/somatic-symptom-and-related-disorders-ssrds#:~:text=It%20is%20important%20to%20note,time%20to%20start%20appropriate%20treatment)). From a data best-practice perspective, reducing detailed lab data into a summary feature per patient (while possibly losing some granularity) greatly simplifies analysis and is common in predictive modeling with EHR data. We ensure this is done carefully by using each lab’s own interpretation flag – a reliable method noted in the MIMIC-III database documentation, where each lab observation comes with an “abnormal” indicator for precisely this kind of use ([MIMIC Observation Labevents \- Definitions \- MIMIC Implementation Guide v1.3.0](https://mimic.mit.edu/fhir/StructureDefinition-mimic-observation-labevents-definitions.html#:~:text=Historically%20used%20for%20laboratory%20results,flow%20sheets%20to%20signal%20the)). Thus, our patient table will include a column like `any_abnormal_lab` (or conversely `all_labs_normal`), derived through proper aggregation of the Lab table.

**Step 3: Determine Referral Patterns (Psychiatry vs Other Specialists).** Using the Referral table, we extract referral events for each patient, focusing on two categories: referrals to **Psychiatry** and referrals to **other specialists**. We join the referral data by Patient ID and then order the referrals by date. For each patient, we can compute metrics such as: *the date of first psychiatry referral*, *the date of first other-specialty referral*, and perhaps *the total number of each type*. The core interest is the sequence order – did the patient see a medical specialist before being referred to psych, or vice versa? We can create a derived field, e.g. `referral_order = "PsychFirst"` or `"SpecialistFirst"` (or `"PsychOnly"` if no other referral, etc.), based on comparing those first referral dates. Technically, this involves a self-join or grouping: in pandas one might split the referral dataframe into two (psych vs non-psych), group each by patient to get the earliest date, then merge those results on patient ID. Each patient’s row in the final table will then have something like `first_psych_referral_date` and `first_specialist_referral_date`, from which we determine the order.

*Justification:* Incorporating referral information captures how the care trajectory unfolds for the patient. A patient who goes to multiple specialists and only later to psychiatry might be following the classic pattern of exclusion of physical disease before addressing psychological factors ([Somatic Symptom Disorder \- StatPearls \- NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK532253/#:~:text=Limited%20laboratory%20testing%20is%20recommended,Excessive)). By contrast, an early psychiatry referral might indicate high suspicion of a somatic disorder from the start. This step is built on the principle of **temporal data analysis** in healthcare – we’re leveraging timestamps of events. It’s aligned with real-world dataset usage; for example, a study using a primary care network’s EHR data had to parse over 600k referral records, classify their specialty, and sort them to analyze referral timelines ( [Identifying new referrals from FPs using EMRs \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4196822/#:~:text=The%20CPCSSN%202012%20data%20contained,outgoing%E2%80%94referrals%20from%20incoming%20consultation%20reports) ). We are doing similarly, albeit focusing on psychiatry vs others. Ensuring we correctly identify “psychiatry” referrals (perhaps by a specialty code or department name in the referral data) is important – misclassification here could skew the sequence. Literature on referral data analysis emphasizes cleaning the specialty field and excluding non-physician referrals ( [Identifying new referrals from FPs using EMRs \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4196822/#:~:text=Ascertaining%20the%20new%20referrals%20to,outgoing%E2%80%94referrals%20from%20incoming%20consultation%20reports) ), which we will follow. The result is that our patient-level table gains columns for *referral presence and order*, e.g., a boolean for “referred to psych at all?”, and an categorical field for the sequence order. This richly describes the patient’s path.

**Step 4: Merging and Final Patient-Level Table Construction.** Once we have the pieces above (NYD flag from diagnoses, lab summary, referral info), we merge them on the patient identifier to form a single table where each row is one patient. In practice, this means using the patient’s unique ID as the key. All our source tables should contain this ID (EncounterDiagnosis and Lab likely have encounter-level entries, but we can still group or mark by patient). We have to be mindful of one-to-many relationships: one patient can have many encounters, labs, referrals. Our prior steps already reduced those to patient-level summaries (any NYD, any abnormal labs, first referral dates). Now we perform left-joins or groupby aggregations to bring these together. The conceptual Python implementation might look like:

\# Pseudocode for merging patient-level data:  
patients \= pd.DataFrame({'PatientID': unique\_patient\_list})  \# base frame of patients of interest

\# NYD flag (from diagnosis codes)  
nyd\_flag \= encounter\_diagnoses\_df.assign(NYD \= encounter\_diagnoses\_df\['ICD9'\].isin(NYD\_codes)) \\  
                                  .groupby('PatientID')\['NYD'\].max()  \# True if any NYD code  
patients \= patients.merge(nyd\_flag, on='PatientID', how='left')

\# Lab summary (any abnormal flag)  
lab\_flag \= lab\_df.groupby('PatientID')\['AbnormalFlag'\].max()  \# assume AbnormalFlag is True for abnormal result  
patients \= patients.merge(lab\_flag, on='PatientID', how='left')  
patients\['All\_labs\_normal'\] \= \~patients\['AbnormalFlag'\]  \# invert if needed

\# Referral dates  
first\_psych \= referral\_df\[referral\_df\['Specialty'\]=="Psychiatry"\].groupby('PatientID')\['ReferralDate'\].min()  
first\_other= referral\_df\[referral\_df\['Specialty'\]\!="Psychiatry"\].groupby('PatientID')\['ReferralDate'\].min()  
patients \= patients.merge(first\_psych, on='PatientID', how='left') \\  
                   .merge(first\_other, on='PatientID', how='left')  
\# Determine referral order  
patients\['Referral\_order'\] \= patients.apply(determine\_order, axis=1)  \# custom logic to compare dates

In this pseudo-code, we create a base list of patients (perhaps those who had NYD or who are being studied), then merge in each piece of derived information. The result is a wide table with columns like `NYD`, `All_labs_normal`, `FirstPsychReferralDate`, `FirstOtherReferralDate`, `Referral_order`. (In actual implementation, one would also handle missing values – e.g., if a patient never had a psych referral, `FirstPsychReferralDate` might be null, and `Referral_order` logic should account for that.)

*Justification:* Merging on patient ID in this manner is a standard practice in healthcare analytics. We ensure a one-row-per-patient structure that is easier to analyze for patient-level outcomes. This approach mirrors the design of many research data warehouses and common data models. For example, the **i2b2 framework** and others aggregate clinical data by patient, linking tables like diagnoses, labs, and procedures via patient (and encounter) identifiers ([i2b2 » OneFlorida+ Clinical Research Network](https://onefl.net/front-door/onefli2b2/#:~:text=OneFlorida%2B%20i2b2%20houses%20patient%20demographics%2C,administration%20for%20all%20OneFlorida%2B%20partners)). By following this, we maintain referential integrity – each data element attached to the correct patient. Every linkage step we performed is grounded in clinical logic (e.g. linking labs to patient by ID, or even by encounter date, ensures that normal/abnormal lab indicator truly reflects that patient’s results). We also justify each feature: the NYD flag captures diagnostic uncertainty (an important feature for our analysis), lab results summarize objective findings (or lack thereof) for the patient, and referral order encapsulates the care pathway. Each of these was derived using domain knowledge and best practices (like using abnormal flags, using earliest referral dates), so when we assemble them, the final dataset is both **well-normalized and clinically meaningful**.

Finally, from an informatics perspective, constructing such a patient-level table is part of the **data preparation best practices** for analysis or machine learning on EHR data. It creates a “tidy” dataset where each patient is an observation with various features that can be used for statistical analysis. We’ve justified each step with literature or standards (coding guidelines for NYD, lab flag usage, referral patterns research) to ensure the method stands on solid ground. This conceptual framework can be implemented in Python (using libraries like pandas for merging and grouping) to reproducibly generate the dataset needed for further analysis. The result will enable us to, say, identify how many NYD patients ended up with all-normal labs and a psychiatry referral – a pattern consistent with somatic symptom disorder – backed by the integrated data we assembled ([Somatic Symptom and Related Disorders (SSRDs) | Children's Hospital of Philadelphia](https://www.chop.edu/conditions-diseases/somatic-symptom-and-related-disorders-ssrds#:~:text=It%20is%20important%20to%20note,time%20to%20start%20appropriate%20treatment)) ([Somatic Symptom Disorder \- StatPearls \- NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK532253/#:~:text=Limited%20laboratory%20testing%20is%20recommended,Excessive)).

**Sources:**

1. CDC / NCHS – NAMCS Coding Instructions (ICD-9 code 799.9 for deferred diagnosis) ([Vital and Health Statistics; Series 2, No. 104 (7/87)](https://www.cdc.gov/nchs/data/series/sr_02/sr02_104.pdf#:~:text=of%20diagnosis%20deferred%20or%20for,on%20special%20conditions%20and%20examinations))  
2. LA County DMH – Coding Bulletin (DSM-IV to ICD-9: Axis I “No Diagnosis” \= V71.09) ( )  
3. CMS & AHA – ICD-9-CM Official Coding Guidelines (re: not coding uncertain diagnoses in outpatient) ([Proper Diagnosis Under ICD-10-CM: Don’t be a Coding ‘Minimalist’](https://www.hmpgloballearningnetwork.com/site/twc/articles/proper-diagnosis-under-icd-10-cm-dont-be-coding-minimalist#:~:text=first,Symptoms%2C%20Signs%2C%20and%20Abnormal%20Clinical))  
4. Ryan *et al.* (2013) – Study on extracting referrals from EMRs (specialty classification & temporal order) ( [Identifying new referrals from FPs using EMRs \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4196822/#:~:text=The%20CPCSSN%202012%20data%20contained,outgoing%E2%80%94referrals%20from%20incoming%20consultation%20reports) )  
5. StatPearls (2023) – *Somatic Symptom Disorder* (notes patients often have exhaustive workups with normal results) ([Somatic Symptom Disorder \- StatPearls \- NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK532253/#:~:text=Limited%20laboratory%20testing%20is%20recommended,Excessive))  
6. Children’s Hospital of Philadelphia – Somatic Symptom Disorders overview (emphasizing normal test results) ([Somatic Symptom and Related Disorders (SSRDs) | Children's Hospital of Philadelphia](https://www.chop.edu/conditions-diseases/somatic-symptom-and-related-disorders-ssrds#:~:text=Diagnosis%20begins%20by%20listening%20carefully,careful%20review%20of%20medical%20testing))  
7. MIMIC-III Clinical Database Guide – Laboratory data abnormal flag usage (indicating high/low/normal results) ([MIMIC Observation Labevents \- Definitions \- MIMIC Implementation Guide v1.3.0](https://mimic.mit.edu/fhir/StructureDefinition-mimic-observation-labevents-definitions.html#:~:text=Historically%20used%20for%20laboratory%20results,flow%20sheets%20to%20signal%20the))  
8. OneFlorida+ Data Trust – i2b2 data integration (aggregating encounters, diagnoses, labs by patient)

---

## **Transition Between Stage 1 & Stage 2**

### **1\. “Not Yet Diagnosed” (NYD) Exploration**

**ICD-9** references:

* `799` (Other unknown and unspecified causes)  
* `V71` (Observation for suspected conditions)

| Preliminary NYD Analysis | Count | % of Diagnoses |
| ----- | ----- | ----- |
| Code 799 patterns | 184,927 | 1.48% |
| → Unique patients | 64,120 | \-- |
| Code V71 patterns | 654 | 0.01% |
| → Unique patients | 575 | \-- |

Text-based patterns (e.g., `\bundiagnosed\b`) also found \~1,359 entries.

**Goal**: Flag patients with repeated NYD/“deferred diagnosis” visits, fueling further analysis of unresolved complaints.

---

### **2\. Referral Table Focus**

We need to see if **patients were referred** to both:

1. **Psychiatry**  
2. Another specialty (e.g., for an undiagnosed pain or body part issue)

**Hypothesis**: If the specialist found no organic cause, the primary care doctor might then refer the patient to mental health.

**Proposed Variables** in the table:

| Variable | Description |
| ----- | ----- |
| PID | Patient ID |
| Age | In years |
| Sex | M/F/etc. |
| NYD (y/n) | Flag if “not yet diagnosed” codes appear |
| Body Part Code | Which system/specialty the referral was for |
| Referred to Psych (y/n) | Whether the patient had a psychiatric referral |
| Other Referral (y/n) | Whether the patient also had other specialist referrals |
| SSD (1/0) | Indication of somatic/somatoform disorder |
| Number of Specialist Referrals | Count of distinct referrals in timeframe |

---

### **3\. Rough Causal Chain**

NYD   
 → (≥3 Normal Labs in 12 months)   
   → Specialist for Body-Part   
     → No Organic Diagnosis   
       → Anxiety(1/0)   
         → Psychiatrist   
           → SSD(1/0)

**Interpretation**:

1. Patient repeatedly coded as undiagnosed →  
2. They accumulate multiple normal lab tests →  
3. They’re referred to a body-part specialist who finds no clear cause →  
4. Anxiety escalates →  
5. Psychiatric referral occurs →  
6. Ultimately flagged as SSD or somatoform.

---

### **4\. Supporting Data Points (From Stage 1\)**

**Diagnostic Patterns (ICD-9)**

* \~8.13M diagnoses used ICD-9 (65.2% of total).  
* Codes 780–789 \= “Symptoms, Signs, Ill-defined Conditions”: 6.96% of all diagnoses.

#### **Body System Distribution**

| Body System | \# Codes | \# Patients |
| ----- | ----- | ----- |
| General | 261,846 | 105,133 |
| GI | 176,674 | 81,585 |
| Neuro | 97,335 | 42,654 |
| Cardio | 36,668 | 23,332 |
| Respiratory | 135,582 | 75,542 |
| Musculo | 50,863 | 25,063 |
| Skin | 60,946 | 36,027 |
| Other | 42,632 | 26,579 |

**Patients by \# Body Systems:**

* 0 systems: 39.27%  
* 1 system: 28.94%  
* ≥2 systems: 31.79%

This multi-system symptom pattern is key for potential SSD.

---

### **5\. Lab Results & Linkage**

**Time-Based Linking**:

* 8.53M total labs → 6.50M (77.7%) linked to encounters.  
* \~1.92M labs (22.3%) remain unlinked (orphaned).

**Normal Lab Threshold**:

* We can define “≥3 normal labs” within 12 months using these linked results.  
* \~48.9% labs are classifiable as normal/abnormal with existing reference range or methods.

---

### **6\. Putting It Together for Stage 2**

1. Identify patients with **NYD** codes.  
2. Check if they have **≥3 normal labs** in 12 months.  
3. See if they got referred for “X” body part without an eventual organic diagnosis.  
4. Check if they have an **Anxiety** flag (ICD-9 300.x or repeated anxiolytics).  
5. Confirm or rule out final SSD, either via explicit code (somatoform) or our multi-criteria approach.  
6. Build a table:  
   * (PID, Age, Sex, NYD, etc.)  
   * This becomes the Stage 2 cohort for analyzing the negative-lab → utilization chain.

---

## **Summary**

* **NYD** helps capture unresolved cases.  
* **Referrals** to psychiatry *vs.* other specialists show potential “pathway” from no-diagnosis to mental health.  
* **≥3 Normal Labs** further flags who might escalate in anxiety/SSD.

**Next Step**:  
 Construct a final Stage 2 dataset with columns for each step in the chain (NYD → labs → referral → anxiety → final SSD). This clarifies the **causal route** we suspect from “not yet diagnosed” to a formal SSD label.

FEEDBACK FROM A CLINICIAN:

Below is a concise summary that only highlights the feedback, suggestions, and future plans discussed during the meeting:

Feedback & Suggestions

	1\.	Include Legacy Criteria and Codes

	•	Because the dataset uses older ICD-9 codes and many clinicians still reference DSM-IV terms, the team should map or integrate these older categories (somatoform, pain disorder, hypochondriasis) alongside DSM-5 criteria.

	2\.	Broaden Inclusion Criteria

	•	Single-System Cases: Even if a patient only has one primary symptom (e.g., chronic pain), they may still meet SSD criteria, so do not limit the scope to multi-system complaints.

	•	Chronic Pain: Look closely at patients with persistent pain issues—even if “medically explained”—because a legitimate condition can coexist with disproportionate anxiety.

	•	Multiple Medication Classes: Go beyond just anxiolytics; consider analgesics, antidepressants, anticonvulsants (e.g., gabapentin for chronic pain), and occasional antipsychotics used off-label (e.g., for severe insomnia). Track usage over six months or more.

	3\.	Leverage Utilization Patterns

	•	High-Frequency Visits: Combine normal lab test patterns with repeated visits (doctor shopping or excessive utilization).

	•	Referrals: Look at specialist referrals (e.g., cardiology, gastroenterology) with inconclusive findings, and whether patients later see psychiatrists or remain undiagnosed.

	4\.	Develop a Severity or Probability Metric

	•	Move away from a simple “in/out” diagnosis. Instead, create a scoring system that factors in repeated negative tests, multiple specialist visits, overlapping meds, and existing comorbidities.

	•	Incorporate variables such as mood/anxiety levels (when documented), prescription patterns, and number of unresolved referrals.

	5\.	Address Comorbidities

	•	Recognize that SSD can coexist with a genuine medical condition (e.g., a fracture, chronic pain, or diabetes). Even where a clear physical cause exists, look for signs of excessive health anxiety or functional impairment.

	6\.	Validation & Refinement

	•	If available, use known SSD diagnoses or physician-labeled cases to validate the model.

	•	Seek additional clinical feedback on ambiguous cases, and refine filters to avoid false positives.

Plans & Next Steps

	•	Refine the Data Filters

	•	Add Pain Medication Analysis: Track chronic prescriptions (e.g., opioids, gabapentin) for ≥6 months.

	•	Expand Symptom Criteria: Capture both single- and multi-system complaints instead of only multi-system.

	•	Maintain Legacy Compatibility: Map ICD-9 and DSM-IV diagnoses to DSM-5 concepts.

	•	Perform Advanced Analytics

	•	Build an SSD Severity Index: Integrate repeated lab findings, medication history, referral patterns, and ICD codes into a continuous score.

	•	Causal Inference Study: Examine whether multiple normal labs (plus other factors) lead to higher healthcare usage, mediated by SSD severity.

	•	Coordinate Offline Discussions

	•	Smaller follow-up meetings (with clinical experts) to finalize data-linking methods, define “chronicity” thresholds, and ensure the model aligns with real-world diagnostic behaviors.

These points focus exclusively on the key feedback, suggestions, and planned action items emerging from the discussion, without reiterating background or rationale.

