import os
import pandas as pd

def load_data(path: str | None = None) -> pd.DataFrame:
    path = path or os.getenv("DATA_PATH", "data/clinical_trial_data.csv")
    if path.startswith("/dbfs/"):
        # Databricks DBFS path is directly accessible
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path)
    # Ensure types
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    return df
