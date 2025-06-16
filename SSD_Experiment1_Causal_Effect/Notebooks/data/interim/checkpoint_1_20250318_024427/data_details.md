Canadian Primary Care Sentinel Surveillance Network (CPCSSN) Database Schema
Overview
The CPCSSN database contains de-identified electronic medical record (EMR) data collected from primary care practices across Canada. This comprehensive healthcare database follows a relational model with tables representing different aspects of patient care, connected through primary and foreign keys. The data is collected in cycles (indicated by Cycle_ID fields), allowing for temporal analysis of healthcare trends.

**IMPORTANT STUDY NOTE**: While CPCSSN encompasses general primary care data, this specific checkpoint contains a subset of patients specifically extracted based on mental health and related diagnostic criteria (ICD-9 codes 290-319, 327, 331-333, 347, 625, 698, 780, 786-788, 799, 995). See SQL query file "00 Mental Health ICD9 Codes Queried from Encounter Diagnosis for Care4Mind Patient Population.sql" for extraction criteria. All patients in this checkpoint have mental health or related somatic symptom presentations.
A notable feature of this database is the standardization process, where many fields have both original values (marked with "_orig" suffix) directly from source EMRs and calculated/standardized values (marked with "_calc" suffix) that have been processed for consistency across different sites and EMR systems.
Primary Tables and Relationships
Patient (Core Entity)
Contains basic demographic information about patients, serving as the central entity to which all other records relate.
Field Name
Data Type
Key Type
Description
Patient_ID
BIGINT
P (Primary)
Unique patient identifier that links across all tables
Sex
Text


Patient's biological sex
BirthYear
Int


Year of birth (used instead of full birthdate for de-identification)
BirthMonth
Int


Month of birth
OptedOut
Text


Flag indicating whether patient has opted out of data collection
OptOutDate
DateTime


Date when patient opted out of data collection

PatientDemographic
Extends the Patient table with additional social determinants of health and demographic details.
Field Name
Data Type
Key Type
Description
PatientDemographic_ID
BIGINT
P (Primary, AutoInt)
Unique identifier for demographic record
Network_ID
BIGINT
F (Foreign)
Network identifier (identifies contributing healthcare network)
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier within the network
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Cycle_ID
Text


Data collection cycle identifier
Occupation
Text


Patient's occupation
HighestEducation
Text


Patient's highest education level attained
HousingStatus
Text


Patient's housing status (e.g., owned, rented, homeless)
ResidencePostalCode
Text


First 3 digits of postal code (for geographic analysis while maintaining privacy)
PatientStatus_orig
Text


Original patient status as recorded in EMR
PatientStatus_calc
Text


Calculated/standardized patient status
Language
Text


Patient's preferred language
Ethnicity
Text


Patient's ethnicity
DeceasedYear
BIGINT


Year of death if applicable
DateCreated
DateTime


Record creation timestamp in the CPCSSN database

Encounter
Represents a clinical visit or interaction between patient and healthcare provider.
Field Name
Data Type
Key Type
Description
Encounter_ID
BIGINT
P (Primary, AutoInt)
Unique encounter identifier (links to many other tables)
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Provider_ID
BIGINT
F (Foreign)
Healthcare provider identifier
Cycle_ID
Text


Data collection cycle identifier
EncounterDate
DateTime


Date and time of the patient-provider encounter
Reason_orig
Text


Original reason for visit as recorded in EMR
Reason_calc
Text


Calculated/standardized reason for visit
EncounterType
Text


Type of encounter (e.g., office visit, phone, virtual)
DateCreated
DateTime


Record creation timestamp

EncounterDiagnosis
Contains diagnoses made during specific encounters, representing point-in-time diagnostic information.
Field Name
Data Type
Key Type
Description
EncounterDiagnosis_ID
BIGINT
P (Primary, AutoInt)
Unique diagnosis identifier
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Encounter_ID
BIGINT
F (Foreign)
Links to related Encounter
Cycle_ID
Text


Data collection cycle identifier
DiagnosisText_orig
Text


Original diagnosis text from EMR
DiagnosisText_calc
Text


Calculated/standardized diagnosis text
DiagnosisCodeType_orig
Text


Original diagnosis code type (e.g., ICD-9, ICD-10)
DiagnosisCodeType_calc
Text


Calculated/standardized diagnosis code type
DiagnosisCode_orig
Text


Original diagnosis code from EMR
DiagnosisCode_calc
Text


Calculated/standardized diagnosis code
DateCreated
DateTime


Record creation timestamp

HealthCondition
Represents ongoing or chronic conditions, distinct from point-in-time encounter diagnoses.
Field Name
Data Type
Key Type
Description
HealthCondition_ID
BIGINT
P (Primary, AutoInt)
Unique health condition identifier
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Encounter_ID
BIGINT
F (Foreign)
Links to Encounter where condition may have been identified
Cycle_ID
Text


Data collection cycle identifier
DiagnosisText_orig
Text


Original diagnosis text from EMR
DiagnosisText_calc
Text


Calculated/standardized diagnosis text
DiagnosisCodeType_orig
Text


Original diagnosis code type
DiagnosisCodeType_calc
Text


Calculated/standardized diagnosis code type
DiagnosisCode_orig
Text


Original diagnosis code from EMR
DiagnosisCode_calc
Text


Calculated/standardized diagnosis code
DateOfOnset
DateTime


Date condition began
SignificantNegativeFlag
Text


Flag for significant negative findings
ActiveInactiveFlag
Text


Whether condition is active or resolved/inactive
DateCreated
DateTime


Record creation timestamp

Medication
Contains detailed information about medications prescribed to patients.
Field Name
Data Type
Key Type
Description
Medication_ID
BIGINT
P (Primary, AutoInt)
Unique medication record identifier
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Encounter_ID
BIGINT
F (Foreign)
Links to the Encounter where medication was prescribed
Cycle_ID
Text


Data collection cycle identifier
StartDate
DateTime


Date medication was started
StopDate
DateTime


Date medication was discontinued (if applicable)
Reason
Text


Clinical reason for medication
DIN
Text


Drug Identification Number (Canadian standard)
Name_orig
Text


Original medication name from EMR
Name_calc
Text


Calculated/standardized medication name
CodeType_orig
Text


Original medication code type (e.g., ATC, DIN)
CodeType_calc
Text


Calculated/standardized medication code type
Code_orig
Text


Original medication code from EMR
Code_calc
Text


Calculated/standardized medication code
Strength
Text


Medication strength (e.g., 500mg)
Dose
Text


Prescribed dose
UnitOfMeasure
Text


Unit of measurement for dose
Frequency
Text


Frequency of administration (e.g., twice daily)
DurationCount
Double


Duration count for prescription
DurationUnit
Text


Duration unit (e.g., days, weeks, months)
DispensedCount
Double


Quantity dispensed
DispensedForm
Text


Form of dispensed medication (e.g., tablet, capsule)
RefillCount
Double


Number of refills authorized
DateCreated
DateTime


Record creation timestamp

Lab
Contains laboratory test results for patients.
Field Name
Data Type
Key Type
Description
Lab_ID
BIGINT
P (Primary, AutoInt)
Unique lab record identifier
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Encounter_ID
BIGINT
F (Foreign)
Links to related Encounter
Cycle_ID
Text


Data collection cycle identifier
PerformedDate
DateTime


Date test was performed
Name_orig
Text


Original test name from EMR
Name_calc
Text


Calculated/standardized test name
CodeType_orig
Text


Original code type
CodeType_calc
Text


Calculated/standardized code type
Code_orig
Text


Original test code from EMR
Code_calc
Text


Calculated/standardized test code
TestResult_orig
Text


Original test result value
TestResult_calc
Text


Calculated/standardized test result value
UpperNormal
Text


Upper limit of normal range
LowerNormal
Text


Lower limit of normal range
NormalRange
Text


Normal range description
UnitOfMeasure_orig
Text


Original unit of measure from EMR
UnitOfMeasure_calc
Text


Calculated/standardized unit of measure
DateCreated
DateTime


Record creation timestamp

MedicalProcedure
Documents medical procedures performed on patients.
Field Name
Data Type
Key Type
Description
MedicalProcedure_ID
BIGINT
P (Primary, AutoInt)
Unique procedure identifier
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Provider_ID
BIGINT
F (Foreign)
Healthcare provider identifier
Cycle_ID
Text


Data collection cycle identifier
PerformedDate
DateTime


Date procedure was performed
Name_orig
Text


Original procedure name from EMR
Name_calc
Text


Calculated/standardized procedure name
DateCreated
DateTime


Record creation timestamp

RiskFactor
Records patient risk factors that could impact health outcomes.
Field Name
Data Type
Key Type
Description
RiskFactor_ID
BIGINT
P (Primary, AutoInt)
Unique risk factor identifier
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Encounter_ID
BIGINT
F (Foreign)
Links to related Encounter
Cycle_ID
Text


Data collection cycle identifier
StartDate
DateTime


Start date of risk factor
EndDate
DateTime


End date of risk factor (if applicable)
Name_orig
Text


Original risk factor name from EMR (e.g., smoking, obesity)
Name_calc
Text


Calculated/standardized risk factor name
Value_orig
Text


Original value from EMR
Value_calc
Text


Calculated/standardized value
Status_orig
Text


Original status from EMR
Status_calc
Text


Calculated/standardized status
Frequency
Text


Frequency of risk factor (e.g., for behaviors)
FrequencyType
Text


Type of frequency
FrequencyUnit
Text


Frequency unit (e.g., daily, weekly)
Duration
Text


Duration of risk factor exposure
DurationType
Text


Type of duration
DurationUnit
Text


Duration unit (e.g., years, months)
EndDuration
Text


End duration
EndDurationType
Text


End duration type
EndDurationUnit
Text


End duration unit
RiskDetails
Text


Additional risk details
DateCreated
DateTime


Record creation timestamp

Referral
Tracks referrals to specialists or healthcare services.
Field Name
Data Type
Key Type
Description
Referral_ID
BIGINT
P (Primary, AutoInt)
Unique referral identifier
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Encounter_ID
BIGINT
F (Foreign)
Links to related Encounter
Cycle_ID
Text


Data collection cycle identifier
CompletedDate
DateTime


Date referral was completed
Name_orig
Text


Original referral name/type/specialty from EMR
Name_calc
Text


Calculated/standardized referral name/type
ConceptCode
Text


Concept code for standardized referral type
DescriptionCode
Text


Description code for standardized referral
DateCreated
DateTime


Record creation timestamp

FamilyHistory
Contains information about health conditions in the patient's family.
Field Name
Data Type
Key Type
Description
FamilyHistory_ID
BIGINT
P (Primary)
Unique family history record identifier
Patient_ID
BIGINT
F (Foreign)
Links to Patient table
Network_ID
BIGINT
F (Foreign)
Network identifier
Site_ID
BIGINT
F (Foreign)
Healthcare site identifier
Encounter_ID
BIGINT
F (Foreign)
Links to related Encounter where history was recorded
Cycle_ID
Text


Data collection cycle identifier
DiagnosisText_orig
Text


Original diagnosis text for family member condition
DiagnosisText_calc
Text


Calculated/standardized diagnosis text
DiagnosisCodeType_orig
Text


Original diagnosis code type
DiagnosisCodeType_calc
Text


Calculated/standardized diagnosis code type
DiagnosisCode_orig
Text


Original diagnosis code
DiagnosisCode_calc
Text


Calculated/standardized diagnosis code
Relationship_orig
Text


Original family relationship as recorded in EMR
Relationship_Side_calc
Text


Calculated family side (maternal/paternal)
Relationship_Degree_calc
Text


Calculated relationship degree (first, second, etc.)
AgeAtOnset
BIGINT


Age when condition began in family member
WasCauseOfDeath
Text


Whether condition caused death of family member
AgeAtDeath
BIGINT


Age at death of family member if applicable
DateCreated
DateTime


Record creation timestamp

Key Database Characteristics
De-identified Structure: All data is de-identified to protect patient privacy while enabling research.
Data Standardization: The "_orig" and "_calc" field pairs throughout the database represent a key feature of CPCSSN's data processing:
"_orig" fields contain raw data as extracted from different EMR systems
"_calc" fields contain standardized values after processing for consistency
Hierarchical Organization: Data follows a hierarchy from Patient → Encounters → Clinical Events (labs, procedures, medications, etc.)
Network and Site Tracking: All tables track the contributing healthcare network and site for data provenance
Temporal Tracking: Data collection cycles (Cycle_ID) allow for temporal analysis and data versioning
Multiple Medical Coding Systems: The database accommodates various medical coding systems (ICD-9, ICD-10, etc.) while standardizing them
