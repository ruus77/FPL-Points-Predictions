import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def train_test_split(df: pd.DataFrame)->dict[str, pd.DataFrame | pd.Series]:
    df = df.copy()
    X = df.drop(columns=["total_points"])
    y = df["total_points"]

    train_mask = (X.season.isin([2223, 2324]))
    valid_mask = (X.season == 2425)
    test_mask = (X.season == 2526)

    return {
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_valid": X[valid_mask], "y_valid": y[valid_mask],
        "X_test": X[test_mask], "y_test": y[test_mask]
    }


class FPLDataPipe:
    def __init__(self, num_cols: list[str], cat_cols: list[str], batch_size: int = 64):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.batch_size = batch_size
        self.preprocessor = self._build_pipeline()
        self.valid_dataloader, self.train_dataloader, self.test_dataloader = None, None, None

    def _build_pipeline(self):
        num_transform = Pipeline([
            ("scaler", MinMaxScaler()),
        ])

        cols_transform = Pipeline([
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_transform, self.num_cols),
            ("col", cols_transform, self.cat_cols)],
            remainder="drop")

        preprocessor.set_output(transform="pandas")
        return preprocessor

    @staticmethod
    def _to_tensor(X, y):
        return (
            torch.tensor(X.to_numpy(), dtype=torch.float32),
            torch.tensor(y.to_numpy().reshape(-1, 1), dtype=torch.float32)
        )

    def prepare_data(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        X_train = self.preprocessor.fit_transform(X_train)

        X_valid = self.preprocessor.transform(X_valid)
        X_test = self.preprocessor.transform(X_test)

        X_train_tensor, y_train_tensor = self._to_tensor(X_train, y_train)
        X_valid_tensor, y_valid_tensor = self._to_tensor(X_valid, y_valid)
        X_test_tensor, y_test_tensor = self._to_tensor(X_test, y_test)

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







