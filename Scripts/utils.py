import pandas as pd
from pathlib import Path
from typing import List

def data_import(season_list: List[str] | None = None) -> pd.DataFrame:
    if season_list is None:
        return pd.DataFrame()

    dfs = []

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_path = project_root / "Fantasy-Premier-League" / "data"

    for season in season_list:
        file_path = data_path / season / "gws" / "merged_gw.csv"

        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            df['season_id'] = season
            dfs.append(df)

        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            print(f"Błąd przy sezonie {season}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, axis=0, ignore_index=True)



def sort_data(df:pd.DataFrame)->pd.DataFrame:
  return df.sort_values(by=["name", "kickoff_time", "GW", "season_id"]).reset_index(drop=True)