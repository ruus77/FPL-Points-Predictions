import pandas as pd
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns


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
            df = pd.read_csv(file_path, encoding='latin-1', low_memory=False).str.decode('utf-8')
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


team_colors = {
    'Arsenal': '#EF0107',
    'Aston Villa': '#95BFE5',
    'Bournemouth': '#DA291C',
    'Brentford': '#E30613',
    'Brighton': '#0057B8',
    'Burnley': '#6C1D45',
    'Chelsea': '#034694',
    'Crystal Palace': '#1B458F',
    'Everton': '#003399',
    'Fulham': '#000000',
    'Ipswich': '#3A64A3',
    'Leeds': '#FFCD00',
    'Leicester': '#0053A0',
    'Liverpool': '#C8102E',
    'Man City': '#6CABDD',
    'Man Utd': '#DA291C',
    'Newcastle': '#241F20',
    "Nott'm Forest": '#DD0000',
    'Southampton': '#D71920',
    'Spurs': '#132257',
    'Sunderland': '#EB172B',
    'West Ham': '#7A263A',
    'Wolves': '#FDB913'
}

def best_players_plots(df: pd.DataFrame,
                       num_players: int,
                       season: str = "2024-25"):
    best_df = (df[df.season_id == season]
               .groupby(["name", "team"])["total_points"].sum()
               .reset_index()
               .sort_values("total_points", ascending=False)
               .head(num_players))

    plt.figure(figsize=(12, 5))
    ax1 = sns.barplot(data=best_df, x="total_points", y="name", hue="team",
                      palette=team_colors, dodge=False, legend=False)

    for container in ax1.containers:
        ax1.bar_label(container)

    plt.title(f"Top {num_players} zawodników - Suma punktów ({season})")
    plt.show()

    plt.figure(figsize=(12, 6))
    line_data = (df[(df.season_id == season) & (df.name.isin(best_df["name"]))]
                 .sort_values(["name", "gw"])
                 .assign(cum_pts=lambda x: x.groupby("name")["total_points"].cumsum()))

    sns.lineplot(data=line_data, x="gw", y="cum_pts", hue="name",
                 palette={n: team_colors.get(t) for n, t in zip(best_df.name, best_df.team)},
                 marker="o")

    plt.title(f"Skumulowana suma punktów {num_players} najlepszych piłkarzy w sezonie ({season})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)


def table_simulation(df: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    if season is None:
        return pd.DataFrame()

    df = df[df['season_id'] == season].copy()

    matches = df.drop_duplicates(subset=['fixture', 'team']).copy()

    conditions = [
        ((matches['was_home'] == True) & (matches['team_h_score'] > matches['team_a_score'])) |
        ((matches['was_home'] == False) & (matches['team_a_score'] > matches['team_h_score'])),
        (matches['team_h_score'] == matches['team_a_score'])
    ]

    choices = [3, 1]

    matches['points'] = np.select(conditions, choices, default=0)

    table = matches.groupby('team')['points'].sum().sort_values(ascending=False)

    return table.reset_index()