# data_utils/processing.py
import pandas as pd

def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    # Obsługa obu wariantów wielkości liter w nazwie kolumny GW
    gw_col = next((c for c in ["GW", "gw"] if c in df.columns), None)
    if gw_col:
        return df.sort_values(by=["name", "kickoff_time", gw_col, "season_id"]).reset_index(drop=True)
    return df