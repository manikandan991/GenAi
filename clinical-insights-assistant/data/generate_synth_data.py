# Databricks notebook source
import pandas as pd
import numpy as np
from pathlib import Path
import os

np.random.seed(42)

num_patients = 200
days_per_patient = 30
patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, num_patients + 1)]
cohorts = ['A', 'B']

records = []
for pid in patient_ids:
    cohort = np.random.choice(cohorts)
    for day in range(1, days_per_patient + 1):
        dosage = np.random.choice([50, 75, 100])
        compliance = np.clip(np.random.normal(90, 10), 50, 100)
        adverse_event = np.random.choice([0, 1], p=[0.9, 0.1])
        base_score = 80 + (dosage - 50) * 0.2 + (compliance - 90) * 0.3 - adverse_event * 15
        outcome = np.clip(np.random.normal(base_score, 5), 40, 100)
        notes_templates = [
            "Patient stable, no complaints.",
            "Mild headache reported, advised rest.",
            "Fatigue noted, monitoring ongoing.",
            "Symptoms improving with current dosage.",
            "Adverse reaction observed, dosage adjustment needed."
        ]
        notes = np.random.choice(notes_templates, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        visit_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=day-1)
        records.append([pid, day, dosage, compliance, adverse_event, notes, outcome, cohort, visit_date])

df = pd.DataFrame(records, columns=[
    'patient_id','trial_day','dosage_mg','compliance_pct',
    'adverse_event_flag','doctor_notes','outcome_score','cohort','visit_date'
])

out = Path("/Workspace/Users/manikandan_nagarasan@epam.com/clinical-insights-assistant/data/clinical_trial_data.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
