import pandas as pd

def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["player_code", "seasson", "gw", "match_id"]).reset_index(drop=True)
