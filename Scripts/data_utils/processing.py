import pandas as pd
from data_utils import colors_config

def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["code", "kickoff_time", "season", "gw"]).reset_index(drop=True)



class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cols_map = colors_config.FEATURES_GROUP

    def ema(self, window_size: int = 4) -> tuple[pd.DataFrame, list[str]]:
        df = self.df.copy()
        if self.cols_map is None:
            return pd.DataFrame(), []

        ema_features = self.cols_map["fpl_cols"] + self.cols_map["perf_cols"] + self.cols_map["target"]

        df = sort_data(df)

        ema_frame = df.groupby(["code", "season"])[ema_features].transform(
            lambda x: x.ewm(span=window_size, adjust=False).mean().shift(1)
        )

        new_cols = [f"{c}_ema_{window_size}" for c in ema_features]
        df[new_cols] = ema_frame.fillna(0)

        return df, new_cols

    def lag(self, lag_size: int = 1) -> tuple[pd.DataFrame, list[str]]:
        df = self.df.copy()
        if self.cols_map is None:
            return pd.DataFrame(), []

        lag_features = self.cols_map["fpl_cols"] + self.cols_map["perf_cols"] + self.cols_map["target"]

        df = sort_data(df)

        lagged_frame = df.groupby(["code", "season"])[lag_features].transform(
            lambda x: x.shift(lag_size)
        )

        new_cols = [f"{c}_lagged_{lag_size}" for c in lag_features]
        df[new_cols] = lagged_frame.fillna(0)

        return df, new_cols

    def features_integration(self, lag_size: int = 1, window_size: int = 8) -> pd.DataFrame:
        df = self.df.copy()
        df = sort_data(df)

        df_ema, ema_cols = self.ema(window_size=window_size)
        df_lag, lag_cols = self.lag(lag_size=lag_size)

        pre_game_cols = list(set(self.cols_map.get("pre_game_cols", [])))

        return pd.concat([df[pre_game_cols], df_ema[ema_cols], df_lag[lag_cols]], axis=1)