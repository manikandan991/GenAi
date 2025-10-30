import pandas as pd

def flag_non_compliance(df: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
    return df[df['compliance_pct'] < threshold].copy()

AE_KEYWORDS = ["headache","fatigue","adverse","reaction","nausea","rash","dizziness"]

def extract_adverse_events(df: pd.DataFrame) -> pd.DataFrame:
    notes = df['doctor_notes'].str.lower().fillna("")
    mask_flag = (df['adverse_event_flag'] == 1)
    mask_kw = notes.apply(lambda x: any(k in x for k in AE_KEYWORDS))
    return df[mask_flag | mask_kw].copy()
