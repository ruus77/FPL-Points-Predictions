import pandas as pd


class MLPSplits:
    def __init__(self, df):
        self.df = df.copy()
        self.df = self._assign_clusters_to_df(self.df)
        self.all_splits = self._generate_all_cluster_splits()

    @staticmethod
    def _assign_clusters_to_df(df: pd.DataFrame) -> pd.DataFrame:
        def cluster_logic(price):
            if price < 5.5:
                return 'Budget'
            elif price <= 7.5:
                return 'Mid'
            else:
                return 'Premium'

        df['cluster'] = df['now_cost_lagged_1'].apply(cluster_logic)
        return df

    @staticmethod
    def _train_test_split(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"X_train": pd.DataFrame(), "y_train": pd.Series()}

        X = df.drop(columns=["event_points"])
        y = df["event_points"]

        is_2425 = (X.season_id == 2425)
        is_2526 = (X.season_id == 2526)

        # Wyznaczamy bieżący GW dla sezonu 2526
        data_2526 = X[is_2526]
        if not data_2526.empty:
            curr_gw = data_2526.gw.max()
            split_gw = curr_gw // 2

            train_mask = is_2425
            valid_mask = is_2526 & (X.gw <= split_gw)
            test_mask = is_2526 & (X.gw > split_gw)
        else:
            train_mask = is_2425
            valid_mask = pd.Series(False, index=X.index)
            test_mask = pd.Series(False, index=X.index)

        return {
            "X_train": X[train_mask], "y_train": y[train_mask],
            "X_valid": X[valid_mask], "y_valid": y[valid_mask],
            "X_test": X[test_mask], "y_test": y[test_mask]
        }

    def _generate_all_cluster_splits(self):
        results = {}
        for cluster_name in ["Budget", "Mid", "Premium"]:
            cluster_data = self.df[self.df.cluster == cluster_name]
            results[cluster_name] = self._train_test_split(cluster_data)
        return results

    def get_splits(self, cluster_name: str):
        return self.all_splits.get(cluster_name)