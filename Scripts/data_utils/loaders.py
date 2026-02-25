import pandas as pd
from pathlib import Path

def vastaav_data_import(season_list: list[str] | None = None) -> pd.DataFrame:
    if season_list is None:
        return pd.DataFrame()

    dfs = []
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


def data_import() -> pd.DataFrame:
    seasons_config = {
        "2024-2025": {
            "match": "https://github.com/olbauday/FPL-Core-Insights/raw/main/data/2024-2025/playermatchstats/GW{gw}/playermatchstats.csv",
            "player": "https://github.com/olbauday/FPL-Core-Insights/raw/main/data/2024-2025/playerstats/playerstats.csv",
            "names": "https://github.com/olbauday/FPL-Core-Insights/raw/main/data/2024-2025/players/players.csv",
            "teams": "https://github.com/olbauday/FPL-Core-Insights/raw/main/data/2024-2025/teams/teams.csv"
        },
        "2025-2026": {
            "match": "https://github.com/olbauday/FPL-Core-Insights/raw/main/data/2025-2026/By%20Gameweek/GW{gw}/playermatchstats.csv",
            "player": "https://github.com/olbauday/FPL-Core-Insights/raw/main/data/2025-2026/By%20Gameweek/GW{gw}/player_gameweek_stats.csv",
            "names": "https://github.com/olbauday/FPL-Core-Insights/raw/main/data/2025-2026/players.csv",
            "teams": "https://github.com/olbauday/FPL-Core-Insights/raw/main/data/2025-2026/teams.csv"
        }
    }

    final_seasons_list = []

    for season, paths in seasons_config.items():
        gw_accumulated = []
        for gw in range(1, 39):
            try:
                df_m = pd.read_csv(paths["match"].format(gw=gw), low_memory=False).assign(season=season, gw=gw)
                df_m = df_m.rename(columns={"player_id": "id"})

                if "{gw}" in paths["player"]:
                    df_p = pd.read_csv(paths["player"].format(gw=gw), low_memory=False).assign(season=season, gw=gw)
                    gw_accumulated.append(pd.merge(df_m, df_p, on=["id", "season", "gw"], suffixes=('', '_dup')))
                else:
                    gw_accumulated.append(df_m)
            except Exception:
                continue

        if not gw_accumulated:
            continue

        df_season = pd.concat(gw_accumulated, ignore_index=True)

        if "{gw}" not in paths["player"]:
            df_player_bulk = pd.read_csv(paths["player"]).assign(season=season)
            df_season = pd.merge(df_season, df_player_bulk, on=["id", "season", "gw"], suffixes=('', '_dup'))

        df_names = pd.read_csv(paths["names"]).assign(season=season).rename(columns={"player_id": "id"})
        cols_to_drop = [c for c in df_names.columns if c in df_season.columns and c not in ["id", "season"]]
        df_season = pd.merge(df_season, df_names.drop(columns=cols_to_drop), on=["id", "season"])

        try:
            df_teams = pd.read_csv(paths["teams"])[["code", "name", "strength"]]
            df_teams = df_teams.rename(columns={"name": "team_name", "strength": "team_strength"})

            df_season = pd.merge(
                df_season,
                df_teams,
                left_on="team_code",
                right_on="code",
                how="left"
            ).drop(columns=["code"])
        except Exception as e:
            print(f"Błąd przy dodawaniu drużyn dla {season}: {e}")

        final_seasons_list.append(df_season)

    common_cols = list(set.intersection(*(set(df.columns) for df in final_seasons_list)))
    common_cols = [c for c in common_cols if not c.endswith('_dup')]
    df = pd.concat([df[common_cols] for df in final_seasons_list], ignore_index=True).convert_dtypes()
    return pd.concat([df[df.select_dtypes(["object", "string"]).columns],
                      df[df.select_dtypes(exclude=["object", "string"]).columns]], axis=1)
