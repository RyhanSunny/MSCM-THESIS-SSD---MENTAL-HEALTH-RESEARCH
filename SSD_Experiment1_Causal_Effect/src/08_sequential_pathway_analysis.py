#!/usr/bin/env python3
"""
Sequential Pathway Analysis for Somatic Symptom Disorder (SSD)

Implements Dr. Felipe's sequential causal chain:
NYD → Normal Labs → Specialist → No Diagnosis → Anxiety → Psychiatrist → SSD
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SSDSequentialAnalyzer:
    """Detects whether patients follow the full NYD→SSD diagnostic pathway."""

    def __init__(self, cohort: pd.DataFrame, health_condition: pd.DataFrame, lab: pd.DataFrame,
                 referral: pd.DataFrame, exposure: pd.DataFrame) -> None:
        self.cohort = cohort
        self.health_condition = health_condition
        self.lab = lab
        self.referral = referral
        self.exposure = exposure

        # Temporal windows in months
        self.pathway_window = 24  # complete pathway
        self.lab_window = 12      # normal labs after NYD
        self.referral_window = 18 # specialist referrals

        self.stages = [
            "nyd",
            "normal_labs",
            "specialist",
            "no_diagnosis",
            "anxiety",
            "psychiatrist",
            "ssd",
        ]

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def build_patient_timeline(self, patient_id: int) -> dict[str, pd.DataFrame]:
        """Return all records for a given patient sorted by date."""
        return {
            "conditions": self.health_condition[self.health_condition["Patient_ID"] == patient_id]
            .sort_values("Date"),
            "labs": self.lab[self.lab["Patient_ID"] == patient_id].sort_values("Date"),
            "referrals": self.referral[self.referral["Patient_ID"] == patient_id].sort_values("Date"),
            "exposures": self.exposure[self.exposure["Patient_ID"] == patient_id].sort_values("Date"),
        }

    def get_nyd_diagnoses(self, patient_id: int, timeline: dict[str, pd.DataFrame]):
        """Return NYD diagnosis dates for the patient."""
        return timeline["conditions"][timeline["conditions"]["Condition"] == "NYD"]["Date"].tolist()

    def get_normal_labs_after_nyd(self, patient_id: int, nyd_dates, timeline: dict[str, pd.DataFrame]):
        """Return dates of normal lab results following each NYD diagnosis."""
        labs = []
        for nyd_date in nyd_dates:
            end = nyd_date + timedelta(days=30 * self.lab_window)
            mask = (
                (timeline["labs"]["Date"] >= nyd_date)
                & (timeline["labs"]["Date"] <= end)
                & (timeline["labs"]["Result"] == "normal")
            )
            labs.extend(timeline["labs"][mask]["Date"].tolist())
        return labs

    def get_medical_specialist_referrals(self, patient_id: int, timeline: dict[str, pd.DataFrame]):
        """Return dates of medical specialist referrals."""
        return timeline["referrals"][timeline["referrals"]["Type"] == "medical"]["Date"].tolist()

    def assess_inconclusive_workup(self, patient_id: int, med_referrals, timeline: dict[str, pd.DataFrame]):
        """Return True if no definitive diagnosis after medical referrals."""
        diagnoses = timeline["conditions"][timeline["conditions"]["Condition"] == "Diagnosis"]["Date"]
        return len(diagnoses) == 0

    def detect_anxiety_after_workup(self, patient_id: int, timeline: dict[str, pd.DataFrame]):
        """Return anxiety diagnosis dates after the workup period."""
        return (
            timeline["conditions"][timeline["conditions"]["Condition"] == "Anxiety"]["Date"].tolist()
        )

    def get_psychiatrist_referral(self, patient_id: int, timeline: dict[str, pd.DataFrame]):
        """Return dates of psychiatric referrals."""
        return timeline["referrals"][timeline["referrals"]["Type"] == "psychiatric"]["Date"].tolist()

    def assess_ssd_outcome(self, patient_id: int):
        """Return True if the patient has an SSD diagnosis."""
        return not self.exposure[self.exposure["Patient_ID"] == patient_id].empty

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def create_pathway_result(
        self,
        patient_id: int,
        stage: int,
        complete_pathway: bool = False,
        nyd_to_ssd_days: int | None = None,
        stages_completed=None,
        stage_dates=None,
    ):
        """Build a single patient pathway summary."""
        if stages_completed is None:
            stages_completed = list(range(stage))

        probability = len(stages_completed) / len(self.stages)

        intervals = []
        if stage_dates:
            prev = None
            for step in self.stages:
                d = stage_dates.get(step)
                if d is not None and prev is not None:
                    intervals.append((d - prev).days)
                else:
                    intervals.append(None)
                if d is not None:
                    prev = d

        return {
            "patient_id": patient_id,
            "pathway_stage": stage,
            "complete_pathway": complete_pathway,
            "nyd_to_ssd_days": nyd_to_ssd_days,
            "stages_completed": stages_completed,
            "bottleneck_stage": None if complete_pathway else stage,
            "probability": probability,
            "interval_days": intervals,
        }

    def detect_complete_pathway(self, patient_id: int):
        """Evaluate whether a patient completes the NYD→SSD pathway."""
        timeline = self.build_patient_timeline(patient_id)
        stage_dates = {}

        # Step 1: NYD diagnosis
        nyd_dates = self.get_nyd_diagnoses(patient_id, timeline)
        if not nyd_dates:
            return self.create_pathway_result(patient_id, stage=0, stage_dates=stage_dates)
        stage_dates["nyd"] = nyd_dates[0]

        # Step 2: Normal labs within window
        normal_labs = self.get_normal_labs_after_nyd(patient_id, nyd_dates, timeline)
        if len(normal_labs) < 3:
            return self.create_pathway_result(patient_id, stage=1, stage_dates=stage_dates)
        stage_dates["normal_labs"] = normal_labs[0]

        # Step 3: Medical specialist referrals
        med_refs = self.get_medical_specialist_referrals(patient_id, timeline)
        if not med_refs:
            return self.create_pathway_result(patient_id, stage=2, stage_dates=stage_dates)
        stage_dates["specialist"] = med_refs[0]

        # Step 4: Inconclusive workup
        if not self.assess_inconclusive_workup(patient_id, med_refs, timeline):
            return self.create_pathway_result(patient_id, stage=3, stage_dates=stage_dates)
        stage_dates["no_diagnosis"] = med_refs[-1] if med_refs else None

        # Step 5: Anxiety/depression emergence
        anxiety_dates = self.detect_anxiety_after_workup(patient_id, timeline)
        if not anxiety_dates:
            return self.create_pathway_result(patient_id, stage=4, stage_dates=stage_dates)
        stage_dates["anxiety"] = anxiety_dates[0]

        # Step 6: Psychiatrist referral
        psych_ref = self.get_psychiatrist_referral(patient_id, timeline)
        if not psych_ref:
            return self.create_pathway_result(patient_id, stage=5, stage_dates=stage_dates)
        stage_dates["psychiatrist"] = psych_ref[0]

        # Step 7: SSD diagnosis
        if not self.assess_ssd_outcome(patient_id):
            return self.create_pathway_result(patient_id, stage=6, stage_dates=stage_dates)

        ssd_date = self.exposure[self.exposure["Patient_ID"] == patient_id]["Date"].iloc[0]
        nyd_to_ssd_days = (ssd_date - nyd_dates[0]).days
        stage_dates["ssd"] = ssd_date

        return self.create_pathway_result(
            patient_id,
            stage=7,
            complete_pathway=True,
            nyd_to_ssd_days=nyd_to_ssd_days,
            stages_completed=list(range(7)),
            stage_dates=stage_dates,
        )

    def analyze_cohort(self) -> pd.DataFrame:
        """Analyze all patients and return a dataframe summarizing pathways."""
        results = [self.detect_complete_pathway(pid) for pid in self.cohort["Patient_ID"]]
        return pd.DataFrame(results)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    DERIVED = ROOT / "data_derived"

    cohort = pd.read_parquet(DERIVED / "cohort.parquet")
    health_condition = pd.read_parquet(DERIVED / "health_condition.parquet")
    lab = pd.read_parquet(DERIVED / "lab.parquet")
    referral = pd.read_parquet(DERIVED / "referral.parquet")
    exposure = pd.read_parquet(DERIVED / "exposure.parquet")

    analyzer = SSDSequentialAnalyzer(cohort, health_condition, lab, referral, exposure)
    result_df = analyzer.analyze_cohort()
    output = DERIVED / "sequential_pathways.parquet"
    result_df.to_parquet(output, index=False)
    logger.info(f"Sequential pathway results saved to {output}")

