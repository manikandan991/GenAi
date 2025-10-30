import pandas as pd

def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby('cohort').agg(
        n_records=('patient_id','count'),
        avg_outcome=('outcome_score','mean'),
        avg_compliance=('compliance_pct','mean'),
        adverse_rate=('adverse_event_flag','mean'),
        avg_dosage=('dosage_mg','mean'),
    ).reset_index()
    agg['adverse_rate'] = (agg['adverse_rate'] * 100).round(2)
    return agg

def compare_cohorts(df: pd.DataFrame) -> dict:
    s = cohort_summary(df)
    if len(s) < 2:
        return {"message":"Only one cohort present.", "summary": s.to_dict(orient="records")}
    a = s[s['cohort']=="A"]
    b = s[s['cohort']=="B"]
    return {"summary": s.to_dict(orient="records")}
