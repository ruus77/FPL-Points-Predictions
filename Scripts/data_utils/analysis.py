# data_utils/analysis.py
import pandas as pd
import numpy as np


def table_simulation(df: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    if season is None:
        return pd.DataFrame()

    df = df[df['season_id'] == season].copy()
    matches = df.drop_duplicates(subset=['match_id', 'team_name']).copy()

    conditions = [
        ((matches['was_home'] == True) & (matches['team_h_score'] > matches['team_a_score'])) |
        ((matches['was_home'] == False) & (matches['team_a_score'] > matches['team_h_score'])),
        (matches['team_h_score'] == matches['team_a_score'])
    ]
    choices = [3, 1]
    matches['points'] = np.select(conditions, choices, default=0)

    table = matches.groupby('team')['points'].sum().sort_values(ascending=False)
    return table.reset_index()