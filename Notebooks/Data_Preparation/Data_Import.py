import pandas as pd
pd.set_option("display.max_columns", None)

import numpy as np
import soccerdata as sd
import config

seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]

class FPLData:
    def __init__(self, season_list: list[str]):
        self.url_manager = {
            "vaastav": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/gws/gw{}.csv",
            "fci" : "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2025-2026/By%20Gameweek/GW{}/{}.csv",
            "fci_playerstats" : "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2025-2026/playerstats.csv"
        }

        self.id_manager = {"vaastav": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/players_raw.csv"}
        self.files_manager = ["player_gameweek_stats", "players", "matches", "teams"]
        self.season_list = season_list

    def _get_id_df(self)-> pd.DataFrame:
        frames, id_frames = [], []

        for season in self.season_list:
            for gw in range(1, 39):
                try:
                    url = self.url_manager["vaastav"].format(season, gw)
                    df = pd.read_csv(url, low_memory=False).assign(season=season, gw=gw)
                    if not df.empty:
                        frames.append(df.assign(season=season, gw=gw).dropna(axis=1, how="all"))
                except Exception as e:
                    print(f"Exception for {season}, {gw}: {e}")

            try:
                url_id = self.id_manager["vaastav"].format(season)
                vaastav_id = pd.read_csv(url_id, low_memory=False, usecols=["id", "first_name", "second_name", "code"]).assign(season=season)
                id_frames.append(vaastav_id)
            except Exception as e:
                print(f"Exception for {season}: {e}")

        if not frames or not id_frames:
            return pd.DataFrame()

        vaastav_data = pd.concat(frames, ignore_index=True)
        id_data = pd.concat(id_frames, ignore_index=True)

        id_data["name"] = id_data["first_name"] + " " + id_data["second_name"]
        id_data = id_data.drop(columns=["first_name", "second_name"])

        df = pd.merge(
            vaastav_data, id_data,
            how="inner",
            left_on=["element", "season"],
            right_on=["id", "season"]
        ).drop(columns=["id"])

        if "name_y" in df.columns:
            df.drop(columns=["name_y"], inplace=True)
        if "name_x" in df.columns:
            df.rename(columns={"name_x": "name"}, inplace=True)

        df["value"] = df["value"] / 10
        return df.convert_dtypes()

    def _get_fci_df(self)-> pd.DataFrame:
        try:
            playerstats = pd.read_csv(self.url_manager["fci_playerstats"], low_memory=False)

            url_teams = self.url_manager["fci_playerstats"].replace("playerstats.csv", "teams.csv")
            teams_df = pd.read_csv(url_teams, usecols=["id", "code"])
            teams_df.rename(columns={"id": "team_id", "code": "team_code"}, inplace=True)

        except Exception as e:
            print(f"Exception {e}")
            return pd.DataFrame()
        players_frames = []
        matches_frames = []

        for gw in range(1, 39):
            try:
                url_players = self.url_manager["fci"].format(gw, "players")
                df_p = pd.read_csv(url_players, low_memory=False, usecols=["player_id", "first_name", "second_name", "position", "player_code", "team_code"])
                players_frames.append(df_p)
            except Exception as e:
                print(f"Exception {e}")

            try:
                url_matches = self.url_manager["fci"].format(gw, "matches")
                df_m = pd.read_csv(url_matches, low_memory=False, usecols=["match_id", "gameweek", "kickoff_time", "home_team", "away_team", "home_score", "away_score"])
                matches_frames.append(df_m)
            except Exception as e:
                print(f"Exception {e}")

        if not players_frames or not matches_frames:
            return pd.DataFrame()

        players = pd.concat(players_frames, ignore_index=True).drop_duplicates(subset=["player_id"])

        players["name"] = players["first_name"] + " " + players["second_name"]
        players = players.drop(columns=["first_name", "second_name"])

        players = pd.merge(players, teams_df, on="team_code", how="left")
        matches = pd.concat(matches_frames, ignore_index=True)

        df = pd.merge(playerstats, players, how="left", left_on="id", right_on="player_id")
        df_matches = pd.merge(df, matches, how="left", left_on="gw", right_on="gameweek")

        df_matches = df_matches[
            (df_matches["team_id"] == df_matches["home_team"]) |
            (df_matches["team_id"] == df_matches["away_team"])
        ].copy()

        df_matches["was_home"] = df_matches["team_id"] == df_matches["home_team"]
        df_matches["opponent_team"] = np.where(df_matches["was_home"], df_matches["away_team"], df_matches["home_team"])

        df_matches["transfers_in_event"] = df_matches["transfers_in_event"].fillna(0)
        df_matches["transfers_out_event"] = df_matches["transfers_out_event"].fillna(0)
        df_matches["transfers_balance"] = df_matches["transfers_in_event"] - df_matches["transfers_out_event"]

        rename_dict = {
            "id": "element",
            "ep_this": "xP",
            "match_id": "fixture",
            "now_cost": "value",
            "selected_by_percent": "selected",
            "home_score": "team_h_score",
            "away_score": "team_a_score",
            "event_points": "total_points",
            "player_code": "code",
            "transfers_in_event": "transfers_in",
            "transfers_out_event": "transfers_out"
        }
        df_matches.rename(columns=rename_dict, inplace=True)
        df_matches["round"] = df_matches["gw"]

        target_columns = [
            'name', 'position', 'xP', 'assists', 'bonus', 'bps', 'clean_sheets',
            'creativity', 'element', 'expected_assists',
            'expected_goal_involvements', 'expected_goals',
            'expected_goals_conceded', 'fixture', 'goals_conceded', 'goals_scored',
            'ict_index', 'influence', 'kickoff_time', 'minutes', 'opponent_team',
            'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards',
            'round', 'saves', 'selected', 'starts', 'team_a_score', 'team_h_score',
            'threat', 'total_points', 'transfers_balance', 'transfers_in',
            'transfers_out', 'value', 'was_home', 'yellow_cards', 'gw', 'code'
        ]

        available_columns = [col for col in target_columns if col in df_matches.columns]
        df_final = df_matches[available_columns].copy()

        return df_final.convert_dtypes()

    def get_data(self)-> pd.DataFrame:
        id_df = self._get_id_df()
        fci_df = self._get_fci_df()

        id_df = id_df.loc[:, ~id_df.columns.duplicated()].copy()
        fci_df = fci_df.loc[:, ~fci_df.columns.duplicated()].copy()
        fci_df["season"] = "2025-26"
        fci_df = fci_df[fci_df['fixture'].str.contains('prem', case=False, na=False)]
        common_cols = [c for c in fci_df.columns]
        data = pd.concat([id_df[common_cols], fci_df[common_cols]], axis=0, ignore_index=True).convert_dtypes()

        data["date"]  = pd.to_datetime(data["kickoff_time"], errors="coerce").dt.date

        return data.drop(columns=["fixture", "round"]).dropna()


fpl_importer = FPLData(season_list=seasons)
data = fpl_importer.get_data()


def wide_to_long_fpl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pairs = [
        ('score', 'team_h_score', 'team_a_score')
    ]

    for base, h_col, a_col in pairs:
        if h_col in df.columns and a_col in df.columns:
            df[f'team_{base}'] = np.where(df['was_home'], df[h_col], df[a_col])
            df[f'opp_{base}'] = np.where(df['was_home'], df[a_col], df[h_col])

            df.drop(columns=[h_col, a_col], inplace=True)

    if 'team_score' in df.columns and 'opp_score' in df.columns:
        df['score_diff'] = df['team_score'] - df['opp_score']

    return df

data = wide_to_long_fpl(data)



class UnderstatData:
    def __init__(self, season_list: list[str]):
        self.season_list = season_list
        self.understat = sd.Understat(leagues="ENG-Premier League", seasons=[s[2:] for s in self.season_list])

    def _get_player_stats(self) -> pd.DataFrame:
        understat = self.understat
        player_stats = understat.read_player_match_stats().reset_index()
        return player_stats

    def _get_team_stats(self) -> pd.DataFrame:
        understat = self.understat
        return understat.read_team_match_stats().reset_index()

    def _merge_undestat(self)-> pd.DataFrame:
        player_stats = self._get_player_stats()
        team_stats = self._get_team_stats()
        return pd.merge(player_stats, team_stats[['season_id', 'game_id', 'away_points', "date",
       'away_expected_points', 'away_goals', 'away_xg', 'away_np_xg',
       'away_np_xg_difference', 'away_ppda', 'away_deep_completions',
       'home_points', 'home_expected_points', 'home_goals', 'home_xg',
       'home_np_xg', 'home_np_xg_difference', 'home_ppda',
       'home_deep_completions']],
                          how="left", on=["game_id", "season_id"])

    def get_undestat(self) -> pd.DataFrame:
        df = self._merge_undestat()

        if 'was_home' not in df.columns:
            df['was_home'] = df.apply(
                lambda row: str(row['game']).split(' ', 1)[1].startswith(str(row['team'])), axis=1)
        pairs = [
            ('points', 'home_points', 'away_points'),
            ('expected_points', 'home_expected_points', 'away_expected_points'),
            ('goals', 'home_goals', 'away_goals'),
            ('xg', 'home_xg', 'away_xg'),
            ('np_xg', 'home_np_xg', 'away_np_xg'),
            ('np_xg_difference', 'home_np_xg_difference', 'away_np_xg_difference'),
            ('ppda', 'home_ppda', 'away_ppda'),
            ('deep_completions', 'home_deep_completions', 'away_deep_completions')
        ]

        for base, h_col, a_col in pairs:
            if h_col in df.columns and a_col in df.columns:
                df[f'team_{base}'] = np.where(df['was_home'], df[h_col], df[a_col])
                df[f'opp_{base}'] = np.where(df['was_home'], df[a_col], df[h_col])

                df.drop(columns=[h_col, a_col], inplace=True)

        if 'team_np_xg' in df.columns and 'opp_np_xg' in df.columns:
            df['team_match_np_xg_diff'] = df['team_np_xg'] - df['opp_np_xg']

        if 'team_ppda' in df.columns and 'opp_ppda' in df.columns:
            df['ppda_diff'] = df['team_ppda'] - df['opp_ppda']

        return df

understat = UnderstatData(season_list=seasons)
understat_data = understat.get_undestat()

understat_data.date = understat_data.date.dt.normalize()
data.date = pd.to_datetime(data.date).dt.normalize()


class DataIntegrator:
    def __init__(self):
        self.bridge_url = "https://raw.githubusercontent.com/ChrisMusson/FPL-ID-Map/refs/heads/main/Understat.csv"

    def _get_bridge(self) -> pd.DataFrame:
        try:
            bridge_df = pd.read_csv(self.bridge_url, usecols=["understat", "code"])
            bridge_df = bridge_df.dropna(subset=['understat', 'code'])
            return bridge_df
        except Exception as e:
            print(f"Błąd podczas pobierania mostu: {e}")
            return pd.DataFrame()

    def integrate(self, fpl_data: pd.DataFrame, understat_data: pd.DataFrame) -> pd.DataFrame:
        bridge = self._get_bridge()
        if bridge.empty: return pd.DataFrame()


        understat_with_bridge = pd.merge(
            understat_data[['player_id', 'shots', 'xg_chain', 'xg_buildup', 'key_passes', "date",
                            'was_home', 'team_points', 'opp_points', 'team_expected_points',
       'opp_expected_points', 'team_goals', 'opp_goals', 'team_xg', 'opp_xg',
       'team_np_xg', 'opp_np_xg', 'team_np_xg_difference',
       'opp_np_xg_difference', 'team_ppda', 'opp_ppda',
       'team_deep_completions', 'opp_deep_completions',
       'team_match_np_xg_diff', 'ppda_diff']],
            bridge[['understat', 'code']],
            left_on='player_id',
            right_on='understat',
            how='left'
        )

        df = pd.merge(
            fpl_data,
            understat_with_bridge,
            left_on=['code', 'date'],
            right_on=['code', 'date'],
            how='left',
            suffixes=('', '_understat')
        )

        return df

integrator = DataIntegrator()
data = integrator.integrate(data, understat_data)

def data_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    team_cols = [
        'team_points', 'opp_points', 'team_expected_points', 'opp_expected_points',
        'team_goals', 'opp_goals', 'team_xg', 'opp_xg', 'team_np_xg', 'opp_np_xg',
        'team_np_xg_difference', 'opp_np_xg_difference', 'team_ppda', 'opp_ppda',
        'team_deep_completions', 'opp_deep_completions', 'team_match_np_xg_diff', 'ppda_diff'
    ]

    ind_cols = ['shots', 'xg_chain', 'xg_buildup', 'key_passes']
    id_cols = ['player_id', 'understat']

    group_keys = ['season', 'gw', 'opponent_team', 'was_home']

    for col in team_cols:
        if col in df.columns:
            df[col] = df.groupby(group_keys)[col].transform(lambda x: x.fillna(x.max()))

    for col in ind_cols:
        if col in df.columns:
            df[col] = np.where(df['minutes'] == 0, 0, df[col])
            df[col] = df[col].fillna(0)

    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    available_team_cols = [c for c in team_cols if c in df.columns]
    if available_team_cols:
        df[available_team_cols] = df[available_team_cols].fillna(0)

    return df


data = data_imputation(data)
data = data.drop(columns=["was_home_understat"])


data.drop_duplicates(keep='last', inplace=True)


data.to_parquet(config.TIDY_DATA_PATH,
                engine='pyarrow',
                index=False)