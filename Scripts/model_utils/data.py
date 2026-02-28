import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder

def train_test_split(df: pd.DataFrame)->tuple[pd.DataFrame, pd.Series,
                                               pd.DataFrame, pd.Series,
                                               pd.DataFrame, pd.Series]:
    df = df.copy()
    X = df.drop(columns=["event_points"])
    y = df["event_points"]
    X_train, X_valid, X_test = X[X.season_id == 2425], X[(X.season_id == 2526) & (X.gw <= X.gw.max()//2)], X[(X.season_id == 2526) & (X.gw > X.gw.max()//2)]
    y_train, y_valid, y_test = y[X.season_id == 2425], y[(X.season_id == 2526) & (X.gw <= X.gw.max()//2)], y[(X.season_id == 2526) & (X.gw > X.gw.max()//2)]
    return (X_train, y_train,
            X_valid, y_valid,
            X_test, y_test)



class FPLDataPipe:
    def __init__(self, num_cols: list[str], cat_cols: list[str], batch_size: int = 64, seq_length: int = 5):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.preprocessor = self._build_pipeline()

        self.feature_size = 0
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = None, None, None

    def _build_pipeline(self):
        num_transform = Pipeline([
            ("scaler", RobustScaler())
        ])

        cols_transform = Pipeline([
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_transform, self.num_cols),
            ("cat", cols_transform, self.cat_cols),
            ("passthrough", "passthrough", ["player_code", "season_id"])
        ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")
        return preprocessor

    def _to_seq_tensor(self, X, y):
        X_seq, y_seq = [], []
        X_features = X.drop(columns=["player_code", "season_id"])

        for _, group in X.groupby(["player_code", "season_id"]):
            if len(group) <= self.seq_length:
                continue

            feat_vals = X_features.loc[group.index].values
            target_vals = y.loc[group.index].values

            for i in range(len(group) - self.seq_length):
                X_seq.append(feat_vals[i : i + self.seq_length])
                y_seq.append(target_vals[i + self.seq_length])

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32).reshape(-1, 1)
        return X_tensor, y_tensor

    def prepare_data(self, X_train, X_valid, X_test, y_train, y_valid, y_test):
        X_train = self.preprocessor.fit_transform(X_train)
        X_valid = self.preprocessor.transform(X_valid)
        X_test = self.preprocessor.transform(X_test)
        self.feature_size = X_train.shape[1] - 2

        X_train_tensor, y_train_tensor = self._to_seq_tensor(X_train, y_train)
        X_valid_tensor, y_valid_tensor = self._to_seq_tensor(X_valid, y_valid)
        X_test_tensor, y_test_tensor = self._to_seq_tensor(X_test, y_test)

        self.train_dataloader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.valid_dataloader = DataLoader(
            TensorDataset(X_valid_tensor, y_valid_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )
        self.test_dataloader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            batch_size=self.batch_size,
            shuffle=False
        )

    def get_dataloaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader







