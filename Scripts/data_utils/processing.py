import pandas as pd


def id_cols_detection(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    id_cols = [c for c in df.columns if (
            "code" in c
            or "name" in c
            or "position" in c
            or 'status' in c
            or c == "season"
            or df[c].dtype == 'object'
            or c == "gw"
    ) and c not in ["touches_opposition_box", "offsides"]]

    new_id_cols = [f"{c}_id" for c in id_cols]
    df = df.rename(columns=dict(zip(id_cols, new_id_cols)))

    other_cols = [c for c in df.columns if c not in new_id_cols]
    return df[new_id_cols + other_cols]


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["player_code_id", "season_id", "gw_id", "match_id"]

    return df.sort_values(by=[col for col in sort_cols if col in df.columns]).reset_index(drop=True)

def duplicated_cols(df:pd.DataFrame)->list[str]:
    return df.columns[df.T.duplicated(keep=False)].tolist()
