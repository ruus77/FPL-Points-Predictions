import pandas as pd
pd.set_option("display.max_columns", None)

import numpy as np
import soccerdata as sd
from abc import ABC, abstractmethod
from functools import wraps
import time


seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

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
        """Główny punkt wejścia do pobierania danych."""
        pass
    

class VaastavData(FetchData):
    def __init__(self, season_list: list[str]):
        self.season_list = season_list
        self.url_manager = {
            "vaastav": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/gws/gw{}.csv"
        }
        self.id_manager = {
            "vaastav" : "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{}/players_raw.csv"
        }


    def _fetch_gameweeks(self) -> pd.DataFrame:
        frames = []
        for season in self.season_list[:-1]:
            for gw in range(1, 39):
                try:
                    url = self.url_manager["vaastav"].format(season, gw)
                    df = pd.read_csv(url)
                    if not df.empty:
                        df = df.assign(season=season, gw=gw).dropna(axis=1, how="all")
                        frames.append(df)
                except Exception as e:
                    print(f"Błąd pobierania GW {gw} dla sezonu {season}: {e}")
        
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_ids(self) -> pd.DataFrame:
        id_frames = []
        for season in self.season_list[:-1]:
            try:
                url_id = self.id_manager["vaastav"].format(season)
                df = pd.read_csv(url_id, usecols=["id", "first_name", "second_name", "code"])
                
                df["season"] = season
                df["name"] = df["first_name"] + " " + df["second_name"]
                id_frames.append(df.drop(columns=["first_name", "second_name"]))
            except Exception as e:
                print(f"Błąd pobierania ID dla sezonu {season}: {e}")
        
        return pd.concat(id_frames, ignore_index=True) if id_frames else pd.DataFrame()

    def _merge_and_clean_data(self, gw_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(
            gw_df, id_df,
            how="inner",
            left_on=["element", "season"],
            right_on=["id", "season"]
        ).drop(columns=["id"])

        if "name_y" in df.columns:
            df.drop(columns=["name_y"], inplace=True)
        if "name_x" in df.columns:
            df.rename(columns={"name_x": "name"}, inplace=True)

        if "value" in df.columns:
            df["value"] = df["value"] / 10

        df = df.drop(columns=['opponent_team', 'modified', 'mng_clean_sheets', 'mng_draw', 'mng_goals_scored', 'mng_loss', 'mng_underdog_draw', 'mng_underdog_win', 'mng_win'])
            
        return df
    
    
    def get_data(self) -> pd.DataFrame:
        gw_df = self._fetch_gameweeks()
        id_df = self._fetch_ids()

        if gw_df.empty or id_df.empty:
            return pd.DataFrame()

        merged_df = self._merge_and_clean_data(gw_df, id_df)
        return merged_df.convert_dtypes()


class FCIData(FetchData):
    def __init__(self, season_list):
        self.season_list = season_list[-1]

        self.url_manager = {
            "fci" : "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2025-2026/By%20Gameweek/GW{}/{}.csv",
            "fci_playerstats" : "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2025-2026/playerstats.csv"
        }
        self.target_columns = [
            'name', 'position', 'xP', 'assists', 'bonus', 'bps', 'clean_sheets',
            'creativity', 'element', 'expected_assists',
            'expected_goal_involvements', 'expected_goals',
            'expected_goals_conceded', 'fixture', 'goals_conceded', 'goals_scored',
            'ict_index', 'influence', 'kickoff_time', 'minutes',
            'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards',
            'round', 'saves', 'selected', 'starts', 'team_a_score', 'team_h_score',
            'threat', 'event_points', 'transfers_balance_event', 'transfers_in_event',
            'transfers_out_event', 'value', 'was_home', 'yellow_cards', 'gw', 'code', "team"
        ]
        
        self.files_manager = ["player_gameweek_stats", "players", "matches", "teams"]
        
    def _get_fci_base(self):
        try:
            playerstats = pd.read_csv(self.url_manager["fci_playerstats"], low_memory=False)
            teams_df = pd.read_csv(self.url_manager["fci_playerstats"].replace("playerstats.csv", "teams.csv"), usecols=["id", "code", "name"])
            teams_df.rename(columns={
                "name" : "team"
            }, inplace=True)
            teams_df.rename(columns={"id": "team_id", "code": "team_code"}, inplace=True)
        
            return playerstats, teams_df
        
        except Exception as e:
            print(f"Exception: {e}")
        
            return pd.DataFrame(), pd.DataFrame()
            
    
    def _fetch_gameweeks(self):
        players_frames = []
        matches_frames = []

        for gw in range(1, 39):
            try:
                url_players = self.url_manager["fci"].format(gw, "players")
                df_p = pd.read_csv(url_players, low_memory=False, usecols=["player_id", "first_name", "second_name", "position", "player_code", "team_code"])
                players_frames.append(df_p)

                
                url_matches = self.url_manager["fci"].format(gw, "matches")
                df_m = pd.read_csv(url_matches, low_memory=False, usecols=["match_id", "gameweek", "kickoff_time", "home_team", "away_team", "home_score", "away_score"])
                matches_frames.append(df_m)
                
                
            except Exception as e:
                print(f"Exception {e}")
                continue
            
        if players_frames and  matches_frames:            
            return pd.concat(players_frames, ignore_index=True).drop_duplicates(subset=["player_id"]), pd.concat(matches_frames, ignore_index=True)

        else:
            return pd.DataFrame(), pd.DataFrame()


    def _clean_and_merge_dfs(self):
        players, matches = self._fetch_gameweeks()
        playerstats, teams_df = self._get_fci_base()
        
        players["name"] = players["first_name"] + " " + players["second_name"]
        players = players.drop(columns=["first_name", "second_name"])

        players = pd.merge(players, teams_df, on="team_code", how="left")

        df = pd.merge(playerstats, players, how="left", left_on="id", right_on="player_id")
        
        df_matches = pd.merge(df, matches, how="left", left_on="gw", right_on="gameweek")
        df_matches = df_matches[(df_matches["team_id"] == df_matches["home_team"]) | (df_matches["team_id"] == df_matches["away_team"])].copy()
        
        df_matches["was_home"] = df_matches["team_id"] == df_matches["home_team"]
        df_matches["opponent_team"] = np.where(df_matches["was_home"], df_matches["away_team"], df_matches["home_team"])
        
        for col in ["transfers_in_event", "transfers_out_event"]:
            df_matches[col] = df_matches[col].fillna(0)
        
        df_matches["transfers_balance_event"] = df_matches["transfers_in_event"] - df_matches["transfers_out_event"]

        return df_matches

    
    def _df_prepare(self):
        df = self._clean_and_merge_dfs()
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
        df.rename(columns={
            "transfers_out_event" : "transfers_out",
            "transfers_balance_event" : "transfers_balance",
            "transfers_in_event" : "transfers_in",
            "event_points" : "total_points"
        }, inplace=True)

        return df.assign(season=self.season_list[0])

    def get_data(self):
        return self._df_prepare()

class UnderstatData(FetchData):
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

    def get_data(self) -> pd.DataFrame:
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



class DataIntegrator(FetchData):
    def __init__(self, season_list):
        self.season_list = season_list
        self.fci = FCIData(season_list=self.season_list)
        self.vaastav = VaastavData(season_list=self.season_list)
        self.understat = UnderstatData(season_list=self.season_list)

        self.bridge_url = "https://raw.githubusercontent.com/ChrisMusson/FPL-ID-Map/refs/heads/main/Understat.csv"
    
    
    def _get_bridge(self) -> pd.DataFrame:
        try:
            bridge = pd.read_csv(self.bridge_url, usecols=["understat", "code"])
            bridge = bridge.dropna(subset=['understat', 'code'])
            return bridge

        except Exception as e:
            print(f"Exception: {e}")
            return pd.DataFrame()

        def _get_fpl_df(self):
            df_vaastav = self.vaastav.get_data()
            df_fci = self.fci.get_data()    
            return pd.concat([df_vaastav, df_fci], axis=0, ignore_index=True).convert_dtypes()

        def integrate(self, fpl_data: pd.DataFrame, understat_data: pd.DataFrame) -> pd.DataFrame:
            bridge = self._get_bridge()
            if bridge.empty:
                 return pd.DataFrame()


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


    @timeit
    def get_data(self) -> pd.DataFrame:


        return self.itegrate()



        