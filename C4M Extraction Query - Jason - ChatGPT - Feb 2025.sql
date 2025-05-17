CREATE TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] (
    Patient_ID BIGINT
);

--Cleanup Code ------------------------------------------------------

UPDATE [C4M_HealthCondition]
SET DiagnosisText_orig = REPLACE(REPLACE(REPLACE(DiagnosisText_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_EncounterDiagnosis]
SET DiagnosisText_orig = REPLACE(REPLACE(REPLACE(DiagnosisText_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_Encounter]
SET Reason_orig = REPLACE(REPLACE(REPLACE(Reason_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_MedicalProcedure]
SET DiagnosisText_orig = REPLACE(REPLACE(REPLACE(DiagnosisText_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_FamilyHistory]
SET DiagnosisText_orig = REPLACE(REPLACE(REPLACE(DiagnosisText_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_Lab]
SET Name_orig = REPLACE(REPLACE(REPLACE(Name_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_Medication]
SET Name_orig = REPLACE(REPLACE(REPLACE(Name_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_PatientDemographic]
SET Occupation = REPLACE(REPLACE(REPLACE(Occupation, '|', ''), '\', ''), '"', ''),
HighestEducation = REPLACE(REPLACE(REPLACE(HighestEducation, '|', ''), '\', ''), '"', ''),
HousingStatus = REPLACE(REPLACE(REPLACE(HousingStatus, '|', ''), '\', ''), '"', ''),
[Language] = REPLACE(REPLACE(REPLACE([Language], '|', ''), '\', ''), '"', ''),
Ethnicity = REPLACE(REPLACE(REPLACE(Ethnicity, '|', ''), '\', ''), '"', ''),
PatientStatus_orig = REPLACE(REPLACE(REPLACE(PatientStatus_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_Referral]
SET Name_orig = REPLACE(REPLACE(REPLACE(Name_orig, '|', ''), '\', ''), '"', '');

UPDATE [C4M_RiskFactor]
SET Value_orig = REPLACE(REPLACE(REPLACE(Value_orig, '|', ''), '\', ''), '"', ''),
Status_orig = REPLACE(REPLACE(REPLACE(Status_orig, '|', ''), '\', ''), '"', ''),
RiskDetails = REPLACE(REPLACE(REPLACE(RiskDetails, '|', ''), '\', ''), '"', ''),
Name_orig = REPLACE(REPLACE(REPLACE(Name_orig, '|', ''), '\', ''), '"', '');


EXEC sp_help 'C4M_PatientDemographic';
EXEC sp_help 'PatientDemographic';


SELECT COLUMN_NAME 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'C4M_PatientDemographic';

INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] (Patient_ID)
SELECT DISTINCT Patient_ID
FROM initialdata.initialdata.encdx;

SELECT COUNT(*) AS total_rows
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID]

--Patient-----D------------------------------------------------------------------

SELECT *
INTO  [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient]
FROM  [CPCSSN_Research_2015SRSC51].[dbo].[Patient]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient]
ALTER COLUMN [Sex] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient]
ALTER COLUMN [BirthYear] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient]
ALTER COLUMN [BirthMonth] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient]
ALTER COLUMN [OptedOut] BIT NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient]
ALTER COLUMN [OptOutDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;


INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient] 
(
    Patient_ID,
    Sex,
    BirthYear,
    BirthMonth,
    OptedOut,
    OptOutDate,
    C4M_Patient_ID
)
SELECT 
    pat.[Patient_ID],
    pat.[Sex],
    pat.[BirthYear],
    pat.[BirthMonth],
    pat.[OptedOut],
    pat.[OptOutDate],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[Patient] AS pat
ON unique_pt.Patient_ID = pat.Patient_ID;



--Patient Demographic------D------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[PatientDemographic]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [PatientDemographic_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [Occupation] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [HighestEducation] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [HousingStatus] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [ResidencePostalCode] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [PatientStatus_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [PatientStatus_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [Language] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [Ethnicity] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [DeceasedYear] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;


INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_PatientDemographic] 
(
    PatientDemographic_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Cycle_ID,
    Occupation,
    HighestEducation,
    HousingStatus,
    ResidencePostalCode,
    PatientStatus_orig,
    PatientStatus_calc,
    Language,
    Ethnicity,
    DeceasedYear,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    pd.[PatientDemographic_ID],
    pd.[Network_ID],
    pd.[Site_ID],
    pd.[Patient_ID],
    pd.[Cycle_ID],
    pd.[Occupation],
    pd.[HighestEducation],
    pd.[HousingStatus],
    pd.[ResidencePostalCode],
    pd.[PatientStatus_orig],
    pd.[PatientStatus_calc],
    pd.[Language],
    pd.[Ethnicity],
    pd.[DeceasedYear],
    pd.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[PatientDemographic] AS pd
ON unique_pt.Patient_ID = pd.Patient_ID;


--Encounter-------D----------------------------------------------------------------

SELECT 
    MAX(LEN(Cycle_ID)) AS Cycle_ID_Length,
    MAX(LEN(Reason_orig)) AS Reason_orig_Length,
    MAX(LEN(Reason_calc)) AS Reason_calc_Length,
    MAX(LEN(EncounterType)) AS EncounterType_Length
FROM [CPCSSN_Research_2015SRSC51].[dbo].[Encounter];

SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'C4M_Encounter';


SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[Encounter]
WHERE 1 = 0;


ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [Provider_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [EncounterDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [Reason_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [Reason_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [EncounterType] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;



INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Encounter] 
(
    Encounter_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Provider_ID,
    Cycle_ID,
    EncounterDate,
    Reason_orig,
    Reason_calc,
    EncounterType,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    enc.[Encounter_ID],
    enc.[Network_ID],
    enc.[Site_ID],
    enc.[Patient_ID],
    enc.[Provider_ID],
    enc.[Cycle_ID],
    enc.[EncounterDate],
    enc.[Reason_orig],
    enc.[Reason_calc],
    enc.[EncounterType],
    enc.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[Encounter] AS enc
ON unique_pt.Patient_ID = enc.Patient_ID;


--EncounterDX----D-------------------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[EncounterDiagnosis]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [EncounterDiagnosis_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [DiagnosisText_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [DiagnosisText_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [DiagnosisCodeType_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [DiagnosisCodeType_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [DiagnosisCode_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [DiagnosisCode_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;


INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_EncounterDiagnosis] 
(
    EncounterDiagnosis_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Encounter_ID,
    Cycle_ID,
    DiagnosisText_orig,
    DiagnosisText_calc,
    DiagnosisCodeType_orig,
    DiagnosisCodeType_calc,
    DiagnosisCode_orig,
    DiagnosisCode_calc,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    encdiag.[EncounterDiagnosis_ID],
    encdiag.[Network_ID],
    encdiag.[Site_ID],
    encdiag.[Patient_ID],
    encdiag.[Encounter_ID],
    encdiag.[Cycle_ID],
    encdiag.[DiagnosisText_orig],
    encdiag.[DiagnosisText_calc],
    encdiag.[DiagnosisCodeType_orig],
    encdiag.[DiagnosisCodeType_calc],
    encdiag.[DiagnosisCode_orig],
    encdiag.[DiagnosisCode_calc],
    encdiag.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[EncounterDiagnosis] AS encdiag
ON unique_pt.Patient_ID = encdiag.Patient_ID;


--Health------D-----------------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[HealthCondition]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [HealthCondition_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [DiagnosisText_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [DiagnosisText_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [DiagnosisCodeType_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [DiagnosisCodeType_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [DiagnosisCode_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [DiagnosisCode_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [DateOfOnset] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [SignificantNegativeFlag] BIT NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [ActiveInactiveFlag] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;


INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_HealthCondition] 
(
    HealthCondition_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Encounter_ID,
    Cycle_ID,
    DiagnosisText_orig,
    DiagnosisText_calc,
    DiagnosisCodeType_orig,
    DiagnosisCodeType_calc,
    DiagnosisCode_orig,
    DiagnosisCode_calc,
    DateOfOnset,
    SignificantNegativeFlag,
    ActiveInactiveFlag,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    hc.[HealthCondition_ID],
    hc.[Network_ID],
    hc.[Site_ID],
    hc.[Patient_ID],
    hc.[Encounter_ID],
    hc.[Cycle_ID],
    hc.[DiagnosisText_orig],
    hc.[DiagnosisText_calc],
    hc.[DiagnosisCodeType_orig],
    hc.[DiagnosisCodeType_calc],
    hc.[DiagnosisCode_orig],
    hc.[DiagnosisCode_calc],
    hc.[DateOfOnset],
    hc.[SignificantNegativeFlag],
    hc.[ActiveInactiveFlag],
    hc.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[HealthCondition] AS hc
ON unique_pt.Patient_ID = hc.Patient_ID;


--Lab--------D---------------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[Lab]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Lab_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [PerformedDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Name_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Name_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [CodeType_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [CodeType_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Code_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [Code_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [TestResult_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [TestResult_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [UpperNormal] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [LowerNormal] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [NormalRange] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [UnitOfMeasure_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [UnitOfMeasure_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;


INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Lab] 
(
    Lab_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Encounter_ID,
    Cycle_ID,
    PerformedDate,
    Name_orig,
    Name_calc,
    CodeType_orig,
    CodeType_calc,
    Code_orig,
    Code_calc,
    TestResult_orig,
    TestResult_calc,
    UpperNormal,
    LowerNormal,
    NormalRange,
    UnitOfMeasure_orig,
    UnitOfMeasure_calc,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    lab.[Lab_ID],
    lab.[Network_ID],
    lab.[Site_ID],
    lab.[Patient_ID],
    lab.[Encounter_ID],
    lab.[Cycle_ID],
    lab.[PerformedDate],
    lab.[Name_orig],
    lab.[Name_calc],
    lab.[CodeType_orig],
    lab.[CodeType_calc],
    lab.[Code_orig],
    lab.[Code_calc],
    lab.[TestResult_orig],
    lab.[TestResult_calc],
    lab.[UpperNormal],
    lab.[LowerNormal],
    lab.[NormalRange],
    lab.[UnitOfMeasure_orig],
    lab.[UnitOfMeasure_calc],
    lab.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[Lab] AS lab
ON unique_pt.Patient_ID = lab.Patient_ID;


--Medication--------D---------------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[Medication]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Medication_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [StartDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [StopDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Reason] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [DIN] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Name_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Name_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [CodeType_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [CodeType_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Code_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Code_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Strength] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Dose] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [UnitOfMeasure] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [Frequency] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [DurationCount] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [DurationUnit] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [DispensedCount] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [DispensedForm] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [RefillCount] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;


INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Medication] 
(
    Medication_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Encounter_ID,
    Cycle_ID,
    StartDate,
    StopDate,
    Reason,
    DIN,
    Name_orig,
    Name_calc,
    CodeType_orig,
    CodeType_calc,
    Code_orig,
    Code_calc,
    Strength,
    Dose,
    UnitOfMeasure,
    Frequency,
    DurationCount,
    DurationUnit,
    DispensedCount,
    DispensedForm,
    RefillCount,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    med.[Medication_ID],
    med.[Network_ID],
    med.[Site_ID],
    med.[Patient_ID],
    med.[Encounter_ID],
    med.[Cycle_ID],
    med.[StartDate],
    med.[StopDate],
    med.[Reason],
    med.[DIN],
    med.[Name_orig],
    med.[Name_calc],
    med.[CodeType_orig],
    med.[CodeType_calc],
    med.[Code_orig],
    med.[Code_calc],
    med.[Strength],
    med.[Dose],
    med.[UnitOfMeasure],
    med.[Frequency],
    med.[DurationCount],
    med.[DurationUnit],
    med.[DispensedCount],
    med.[DispensedForm],
    med.[RefillCount],
    med.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[Medication] AS med
ON unique_pt.Patient_ID = med.Patient_ID;


--Medical Procedure------D-----------------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[MedicalProcedure]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [MedicalProcedure_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [PerformedDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [Name_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [Name_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ALTER COLUMN [DateCreated] DATE NULL;


-- Adding a new column
ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;

INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_MedicalProcedure] 
(
    MedicalProcedure_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Encounter_ID,
    Cycle_ID,
    PerformedDate,
    Name_orig,
    Name_calc,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    medproc.[MedicalProcedure_ID],
    medproc.[Network_ID],
    medproc.[Site_ID],
    medproc.[Patient_ID],
    medproc.[Encounter_ID],
    medproc.[Cycle_ID],
    medproc.[PerformedDate],
    medproc.[Name_orig],
    medproc.[Name_calc],
    medproc.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[MedicalProcedure] AS medproc
ON unique_pt.Patient_ID = medproc.Patient_ID;



--Referrals---------D--------------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[Referral]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [Referral_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [CompletedDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [Name_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [Name_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [ConceptCode] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [DescriptionCode] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;


INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Referral] 
(
    Referral_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Encounter_ID,
    Cycle_ID,
    CompletedDate,
    Name_orig,
    Name_calc,
    ConceptCode,
    DescriptionCode,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    ref.[Referral_ID],
    ref.[Network_ID],
    ref.[Site_ID],
    ref.[Patient_ID],
    ref.[Encounter_ID],
    ref.[Cycle_ID],
    ref.[CompletedDate],
    ref.[Name_orig],
    ref.[Name_calc],
    ref.[ConceptCode],
    ref.[DescriptionCode],
    ref.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[Referral] AS ref
ON unique_pt.Patient_ID = ref.Patient_ID;


--Risk Factors------D-----------------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[RiskFactor]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [RiskFactor_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [StartDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [EndDate] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Name_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Name_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Value_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Value_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Status_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Status_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Frequency] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [FrequencyType] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [FrequencyUnit] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [Duration] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [DurationType] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [DurationUnit] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [EndDuration] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [EndDurationType] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [EndDurationUnit] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [RiskDetails] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;



INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_RiskFactor] 
(
    RiskFactor_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Encounter_ID,
    Cycle_ID,
    StartDate,
    EndDate,
    Name_orig,
    Name_calc,
    Value_orig,
    Value_calc,
    Status_orig,
    Status_calc,
    Frequency,
    FrequencyType,
    FrequencyUnit,
    Duration,
    DurationType,
    DurationUnit,
    EndDuration,
    EndDurationType,
    EndDurationUnit,
    RiskDetails,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    rf.[RiskFactor_ID],
    rf.[Network_ID],
    rf.[Site_ID],
    rf.[Patient_ID],
    rf.[Encounter_ID],
    rf.[Cycle_ID],
    rf.[StartDate],
    rf.[EndDate],
    rf.[Name_orig],
    rf.[Name_calc],
    rf.[Value_orig],
    rf.[Value_calc],
    rf.[Status_orig],
    rf.[Status_calc],
    rf.[Frequency],
    rf.[FrequencyType],
    rf.[FrequencyUnit],
    rf.[Duration],
    rf.[DurationType],
    rf.[DurationUnit],
    rf.[EndDuration],
    rf.[EndDurationType],
    rf.[EndDurationUnit],
    rf.[RiskDetails],
    rf.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[RiskFactor] AS rf
ON unique_pt.Patient_ID = rf.Patient_ID;


--Family------D-----------------------------------------------------------------

SELECT *
INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
FROM [CPCSSN_Research_2015SRSC51].[dbo].[FamilyHistory]
WHERE 1 = 0;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [FamilyHistory_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [Network_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [Site_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [Patient_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [Encounter_ID] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [Cycle_ID] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [DiagnosisText_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [DiagnosisText_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [DiagnosisCodeType_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [DiagnosisCodeType_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [DiagnosisCode_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [DiagnosisCode_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [Relationship_orig] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [Relationship_Side_calc] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [Relationship_Degree_calc] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [AgeAtOnset] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [VitalStatus] NVARCHAR(255) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [WasCauseOfDeath] BIT NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [AgeAtDeath] NUMERIC(18, 0) NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ALTER COLUMN [DateCreated] DATE NULL;

ALTER TABLE [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory]
ADD [C4M_Patient_ID] NUMERIC(18, 0) NOT NULL;


INSERT INTO [CPCSSN_Research_2015SRSC51].[dbo].[C4M_FamilyHistory] 
(
    FamilyHistory_ID,
    Network_ID,
    Site_ID,
    Patient_ID,
    Encounter_ID,
    Cycle_ID,
    DiagnosisText_orig,
    DiagnosisText_calc,
    DiagnosisCodeType_orig,
    DiagnosisCodeType_calc,
    DiagnosisCode_orig,
    DiagnosisCode_calc,
    Relationship_orig,
    Relationship_Side_calc,
    Relationship_Degree_calc,
    AgeAtOnset,
    VitalStatus,
    WasCauseOfDeath,
    AgeAtDeath,
    DateCreated,
    C4M_Patient_ID
)
SELECT 
    fh.[FamilyHistory_ID],
    fh.[Network_ID],
    fh.[Site_ID],
    fh.[Patient_ID],
    fh.[Encounter_ID],
    fh.[Cycle_ID],
    fh.[DiagnosisText_orig],
    fh.[DiagnosisText_calc],
    fh.[DiagnosisCodeType_orig],
    fh.[DiagnosisCodeType_calc],
    fh.[DiagnosisCode_orig],
    fh.[DiagnosisCode_calc],
    fh.[Relationship_orig],
    fh.[Relationship_Side_calc],
    fh.[Relationship_Degree_calc],
    fh.[AgeAtOnset],
    fh.[VitalStatus],
    fh.[WasCauseOfDeath],
    fh.[AgeAtDeath],
    fh.[DateCreated],
    unique_pt.[Patient_ID] AS C4M_Patient_ID
FROM [CPCSSN_Research_2015SRSC51].[dbo].[C4M_Patient_ID] AS unique_pt
LEFT JOIN [CPCSSN_Research_2015SRSC51].[dbo].[FamilyHistory] AS fh
ON unique_pt.Patient_ID = fh.Patient_ID;

