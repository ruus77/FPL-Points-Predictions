import pandas as pd
pd.set_option("display.max_columns", None)

import sys
from pathlib import Path

# Dodajemy tylko katalog główny projektu
project_root = str(Path(__file__).resolve().parents[2])

if project_root not in sys.path:
    sys.path.append(project_root)

# Import jawny z podkatalogu
from Scripts import config

import numpy as np
from abc import ABC, abstractmethod
from functools import wraps
import time
import re


seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)

        print(f"Func: {func.__name__} |: {time.time() - start:3f}")

        return result

    return wrapper


class FetchData(ABC):

    def __init__(self, season_list: list):
        self.season_list = season_list
        self.url_manager = {}

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass


class VaastavData(FetchData):

    def __init__(self, season_list: list[str]):

        self.season_list = season_list

        self.url_manager = {
            "vaastav":
            "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/gws/gw{}.csv"
        }

        self.id_manager = {
            "vaastav":
            "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/players_raw.csv"
        }

    def _fetch_gameweeks(self) -> pd.DataFrame:

        frames = []

        for season in self.season_list[:-1]:

            for gw in range(1, 39):

                try:
                    url = self.url_manager["vaastav"].format(
                        season,
                        gw
                    )

                    df = pd.read_csv(url)

                    if not df.empty:
                        df = (
                            df
                            .assign(
                                season=season,
                                gw=gw
                            )
                            .dropna(axis=1, how="all")
                        )

                        frames.append(df)

                except Exception as e:
                    print(
                        f"Błąd pobierania GW {gw} dla sezonu {season}: {e}"
                    )

        return (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame()
        )

    def _fetch_ids(self) -> pd.DataFrame:

        id_frames = []

        for season in self.season_list[:-1]:

            try:
                url_id = self.id_manager["vaastav"].format(season)

                df = pd.read_csv(
                    url_id,
                    usecols=[
                        "id",
                        "first_name",
                        "second_name",
                        "code"
                    ]
                )

                df["season"] = season
                df["name"] = (
                    df["first_name"]
                    + " "
                    + df["second_name"]
                )

                id_frames.append(
                    df.drop(
                        columns=[
                            "first_name",
                            "second_name"
                        ]
                    )
                )

            except Exception as e:
                print(
                    f"Błąd pobierania ID dla sezonu {season}: {e}"
                )

        return (
            pd.concat(id_frames, ignore_index=True)
            if id_frames else pd.DataFrame()
        )

    def _merge_and_clean_data(
        self,
        gw_df: pd.DataFrame,
        id_df: pd.DataFrame
    ) -> pd.DataFrame:

        df = pd.merge(
            gw_df,
            id_df,
            how="inner",
            left_on=["element", "season"],
            right_on=["id", "season"]
        ).drop(columns=["id"])

        if "name_y" in df.columns:
            df.drop(columns=["name_y"], inplace=True)

        if "name_x" in df.columns:
            df.rename(
                columns={"name_x": "name"},
                inplace=True
            )

        if "value" in df.columns:
            df["value"] = df["value"] / 10

        df = df.drop(
            columns=[
                'opponent_team',
                'modified',
                'mng_clean_sheets',
                'mng_draw',
                'mng_goals_scored',
                'mng_loss',
                'mng_underdog_draw',
                'mng_underdog_win',
                'mng_win'
            ],
            errors="ignore"
        )

        return df

    def get_data(self) -> pd.DataFrame:

        gw_df = self._fetch_gameweeks()
        id_df = self._fetch_ids()

        if gw_df.empty or id_df.empty:
            return pd.DataFrame()

        merged_df = self._merge_and_clean_data(
            gw_df,
            id_df
        )

        return merged_df


class FCIData(FetchData):

    def __init__(self, season_list):

        super().__init__([season_list[-1]])

        self.url_manager = {
            "fci":
            "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2025-2026/By%20Gameweek/GW{}/{}.csv",

            "fci_playerstats":
            "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2025-2026/playerstats.csv"
        }

        self.target_columns = [
            'name',
            'position',
            'xP',
            'assists',
            'bonus',
            'bps',
            'clean_sheets',
            'creativity',
            'element',
            'expected_assists',
            'expected_goal_involvements',
            'expected_goals',
            'expected_goals_conceded',
            'fixture',
            'goals_conceded',
            'goals_scored',
            'ict_index',
            'influence',
            'kickoff_time',
            'minutes',
            'own_goals',
            'penalties_missed',
            'penalties_saved',
            'red_cards',
            'round',
            'saves',
            'selected',
            'starts',
            'team_a_score',
            'team_h_score',
            'threat',
            'event_points',
            'transfers_balance_event',
            'transfers_in_event',
            'transfers_out_event',
            'value',
            'was_home',
            'yellow_cards',
            'gw',
            'code',
            'team'
        ]

    def _get_fci_base(self):

        try:
            playerstats = pd.read_csv(
                self.url_manager["fci_playerstats"],
                low_memory=False
            )

            teams_df = pd.read_csv(
                self.url_manager["fci_playerstats"].replace(
                    "playerstats.csv",
                    "teams.csv"
                ),
                usecols=["id", "code", "name"]
            )

            teams_df.rename(
                columns={"name": "team"},
                inplace=True
            )

            teams_df.rename(
                columns={
                    "id": "team_id",
                    "code": "team_code"
                },
                inplace=True
            )

            return playerstats, teams_df

        except Exception as e:
            print(f"Exception: {e}")

            return pd.DataFrame(), pd.DataFrame()

    def _fetch_gameweeks(self):

        players_frames = []
        matches_frames = []

        for gw in range(1, 39):

            try:
                url_players = self.url_manager["fci"].format(
                    gw,
                    "players"
                )

                df_p = pd.read_csv(
                    url_players,
                    low_memory=False,
                    usecols=[
                        "player_id",
                        "first_name",
                        "second_name",
                        "position",
                        "player_code",
                        "team_code"
                    ]
                )

                players_frames.append(df_p)

                url_matches = self.url_manager["fci"].format(
                    gw,
                    "matches"
                )

                df_m = pd.read_csv(
                    url_matches,
                    low_memory=False,
                    usecols=[
                        "match_id",
                        "gameweek",
                        "kickoff_time",
                        "home_team",
                        "away_team",
                        "home_score",
                        "away_score"
                    ]
                )

                matches_frames.append(df_m)

            except Exception as e:
                print(f"Exception {e}")

                continue

        if players_frames and matches_frames:

            return (
                pd.concat(
                    players_frames,
                    ignore_index=True
                ).drop_duplicates(subset=["player_id"]),

                pd.concat(
                    matches_frames,
                    ignore_index=True
                )
            )

        else:
            return pd.DataFrame(), pd.DataFrame()

    def _clean_and_merge_dfs(self):

        players, matches = self._fetch_gameweeks()

        playerstats, teams_df = self._get_fci_base()

        players["name"] = (
            players["first_name"]
            + " "
            + players["second_name"]
        )

        players = players.drop(
            columns=[
                "first_name",
                "second_name"
            ]
        )

        players = pd.merge(
            players,
            teams_df,
            on="team_code",
            how="left"
        )

        df = pd.merge(
            playerstats,
            players,
            how="left",
            left_on="id",
            right_on="player_id"
        )

        df_matches = pd.merge(
            df,
            matches,
            how="left",
            left_on="gw",
            right_on="gameweek"
        )

        df_matches = df_matches[
            (
                df_matches["team_id"]
                == df_matches["home_team"]
            )
            |
            (
                df_matches["team_id"]
                == df_matches["away_team"]
            )
        ].copy()

        df_matches["was_home"] = (
            df_matches["team_id"]
            == df_matches["home_team"]
        )

        for col in [
            "transfers_in_event",
            "transfers_out_event"
        ]:
            df_matches[col] = (
                df_matches[col]
                .fillna(0)
            )

        df_matches["transfers_balance_event"] = (
            df_matches["transfers_in_event"]
            -
            df_matches["transfers_out_event"]
        )

        df_matches["position"] = (
            df_matches["position"]
            .map({
                "Defender": "DEF",
                "Forward": "FWD",
                "Goalkeeper": "GK",
                "Midfielder": "MID"
            })
        )

        return df_matches

    def _df_prepare(self):

        df = self._clean_and_merge_dfs()
        
        df = df[df["match_id"].astype(str).str.contains("prem", na=False)]
        rename_dict = {
            "id": "element",
            "ep_this": "xP",
            "match_id": "fixture",
            "now_cost": "value",
            "selected_by_percent": "selected",
            "home_score": "team_h_score",
            "away_score": "team_a_score",
            "player_code": "code",
        }
        df.rename(columns=rename_dict, inplace=True)

        df["round"] = df["gw"]

        try:
            df = df[self.target_columns].copy()

        except Exception as e:
            print(f"Exception: {e}")

        df.rename(
            columns={
                "transfers_out_event": "transfers_out",
                "transfers_balance_event": "transfers_balance",
                "transfers_in_event": "transfers_in",
                "event_points": "total_points"
            },
            inplace=True
        )

        return df.assign(season="2025-26")

    def get_data(self):

        return self._df_prepare()


class DataIntegrator(FetchData):

    def __init__(self, season_list):

        super().__init__(season_list)

        self.fci = FCIData(
            season_list=self.season_list
        )

        self.vaastav = VaastavData(
            season_list=self.season_list
        )

    def _get_fpl_df(self) -> pd.DataFrame:

        print("Vaastav")
        df_vaastav = self.vaastav.get_data()
        print(f"Vaastav: {df_vaastav.shape}")

        print("FCI")
        df_fci = self.fci.get_data()
        print(f"FCI: {df_fci.shape}")

        df = pd.concat(
            [df_vaastav, df_fci],
            axis=0,
            ignore_index=True
        )

        df['kickoff_time'] = (
            df['kickoff_time']
            .astype(str)
            .str.replace('Z', '', regex=False)
            .str.slice(0, 19)
        )


        df['season'] = df['season'].apply(
            lambda x: re.sub(
                r'^20(\d{2})-(\d{2})$',
                r'\1\2',
                str(x)
            )
        )

        for col in ["team", "season"]:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

        df = df.dropna(
            subset=[
                "team",
                "season",
                "was_home"
            ]
        )
        return df

    @timeit
    def get_data(self) -> pd.DataFrame:

        df = self._get_fpl_df()

        df = df.drop(
            columns=["fixture"],
            errors="ignore"
        )

        return df.dropna(subset=["kickoff_time"])


fpl = DataIntegrator(season_list=seasons)

data = fpl.get_data()

print(data.season.unique())
print(data.shape)

print(
    data.isna().mean()[
        data.isna().mean() > 0
    ]
)

data.to_parquet(
    config.TIDY_DATA_PATH,
    index=False
)