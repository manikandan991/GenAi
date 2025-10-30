import pandas as pd

def simulate_dosage_adjustment(df: pd.DataFrame, delta_mg: int = 25) -> pd.DataFrame:
    sim = df.copy()
    sim['dosage_mg'] = sim['dosage_mg'] + delta_mg
    # naive projection: +0.15 outcome per 25mg, -0.05 adverse risk per 25mg if baseline < 100mg
    sim['projected_outcome'] = (sim['outcome_score'] + (delta_mg/25.0)*0.15* (100 - sim['dosage_mg']).clip(lower=0)/50.0).clip(40,100)
    return sim
