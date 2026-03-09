import pandas as pd

def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["player_code", "season_id", "gw", "match_id"]).reset_index(drop=True)


def id_cols_detection(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in df.columns if (
            ("_id" in c or "id_" in c or c == "id")
            or "code" in c
            or "name" in c
            or "position" in c
            or 'status' in c
            or df[c].dtype == 'object'
            or c == "gw"
    ) or c in ["season_id", "gw"]]
    ic_cols = [f"{c}_id" for c in id_cols if "id" not in c]
    return df[id_cols + [c for c in df.columns if c not in id_cols]]



def duplicated_cols(df:pd.DataFrame)->list[str]:
    return df.columns[df.T.duplicated(keep=False)].tolist()
