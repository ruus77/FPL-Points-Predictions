# data_utils/loaders.py
import pandas as pd
from pathlib import Path

def data_import(season_list: list[str] | None = None) -> pd.DataFrame:
    if season_list is None:
        return pd.DataFrame()

    dfs = []
    # Uwaga: Ścieżka skorygowana o jeden poziom wyżej, bo jesteśmy w folderze data_utils/
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "Fantasy-Premier-League" / "data"

    for season in season_list:
        file_path = data_path / season / "gws" / "merged_gw.csv"
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
            text_cols = df.select_dtypes(include=["object"]).columns
            if len(text_cols) > 0:
                df[text_cols] = df[text_cols].apply(
                    lambda s: s.str.encode("latin-1").str.decode("utf-8", errors="ignore")
                )
            df["season_id"] = season
            dfs.append(df)
        except Exception as e:
            print(f"Błąd przy sezonie {season}: {e}")

    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()

def club_data_import(season_list: list[str]) -> pd.DataFrame:
    url = "https://www.football-data.co.uk/mmz4281"
    dfs = []
    for s in season_list:
        season_code = s[2:4] + s[5:7]
        df = pd.read_csv(f"{url}/{season_code}/E0.csv")
        df['season_id'] = s
        dfs.append(df.copy())
    return pd.concat(dfs, ignore_index=True)