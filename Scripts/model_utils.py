import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class ModelSelector:

    def __init__(self, random_state:int=77, scoring:str="mse"):
      self.random_state = random_state
      self.scoring = scoring

    @staticmethod
    def metrics_report(y_pred: np.ndarray, y_true: np.ndarray):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return mse, mae, r2

    def params_search(self,
                    models: list[sklearn.base.BaseEstimator],
                    models_names: list[str],
                    params_grid: list[dict[str, list[str]]],
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    cv:int=5,
                    scoring:str | None=None,
                    n_iter: int = 20):

        scoring = scoring if scoring else self.scoring

        results_list = []
        best_models_map = {}

        for model, name, grid in zip(models, models_names, params_grid):
            random_search = RandomizedSearchCV(cv=TimeSeriesSplit(n_splits=cv),
                                               n_iter=n_iter,
                                               estimator=model,
                                               scoring=scoring,
                                               param_distributions=grid,
                                               verbose=1,
                                               n_jobs=-1,
                                               error_score="raise",
                                               random_state=self.random_state,
                                               refit=True)
            random_search.fit(X_train, y_train)

            best_models_map[name] = random_search.best_estimator_
            cv_score = random_search.best_score_

            y_train_pred = random_search.best_estimator_.predict(X_train)
            metrics = self.metrics_report(y_true=y_train,
                                        y_pred=y_train_pred)
            row = {
                "data": "train",
                "name": name,
                f"cv_mean_{scoring}" : cv_score,
                "best_params": random_search.best_params_,
                "mse": metrics[0],
                "mae": metrics[1],
                "r2": metrics[2]}

            results_list.append(row)

        return pd.DataFrame(results_list).sort_values(by=f"cv_mean_{scoring}", ascending=False), best_models_map

    def evaluate(self, trained_models_map, X_test, y_test):
        results_list = []
        y_preds = {}

        for name, model in trained_models_map.items():
            y_pred = model.predict(X_test)
            y_preds[name] = y_pred

            metrics = self.metrics_report(y_pred=y_pred, y_true=y_test)

            row = {
                "data": "test",
                "model_name": name,
                "mse": metrics[0],
                "mae": metrics[1],
                "r2": metrics[2]}
            results_list.append(row)

        df_results = pd.DataFrame(results_list)

        sort_column = self.scoring if self.scoring in df_results.columns else "mse"

        return df_results.sort_values(by=sort_column, ascending=False), y_preds




def train_test_split(df: pd.DataFrame)->tuple[pd.DataFrame, pd.Series,
                                               pd.DataFrame, pd.Series,
                                               pd.DataFrame, pd.Series]:
    df = df.copy()
    X = df.drop(columns=["total_points"])
    y = df["total_points"]
    X_train, X_valid, X_test = X[X.season_id < "2024-25"], X[X.season_id == "2024-25"], X[X.season_id > "2024-25"]
    y_train, y_valid, y_test = y[X.season_id < "2024-25"], y[X.season_id == "2024-25"], y[X.season_id > "2024-25"]
    return (X_train, y_train,
            X_valid, y_valid,
            X_test, y_test)



def one_hot_encode(X_train:pd.DataFrame,
                   X_valid:pd.DataFrame,
                   X_test:pd.DataFrame,
                   cols_to_encode:list[str])->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.set_output(transform="pandas")

    enc_train = encoder.fit_transform(X_train[cols_to_encode]).astype(int)
    enc_valid = encoder.transform(X_valid[cols_to_encode]).astype(int)
    enc_test = encoder.transform(X_test[cols_to_encode]).astype(int)

    X_train, X_valid, X_test = (X_train.drop(columns=cols_to_encode),
                                X_valid.drop(columns=cols_to_encode),
                                X_test.drop(columns=cols_to_encode))

    return (pd.concat([X_train, enc_train], axis=1).reset_index(drop=True).select_dtypes(include=np.number),
            pd.concat([X_valid, enc_valid], axis=1).reset_index(drop=True).select_dtypes(include=np.number),
            pd.concat([X_test, enc_test], axis=1).reset_index(drop=True).select_dtypes(include=np.number))


def scale_data(X_train:pd.DataFrame,
               X_valid:pd.DataFrame,
               X_test:pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")

    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return (X_train,
            X_valid,
            X_test)


class FPLDataPipe:
    def __init__(self, num_cols: list[str], cat_cols: list[str], batch_size: int = 64):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.batch_size = batch_size
        self.preprocessor = self._build_pipeline()

    def _build_pipeline(self):
        num_transform = Pipeline([
            ("scaler", MinMaxScaler())
        ])

        cols_transform = Pipeline([
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_transform, self.num_cols),
            ("col", cols_transform, self.cat_cols)],
            remainder="drop")

        preprocessor.set_output(transform="pandas")
        return preprocessor

    def _to_tensor(self, X, y):
        return (
            torch.tensor(X.to_numpy(), dtype=torch.float32),
            torch.tensor(y.to_numpy().reshape(-1, 1), dtype=torch.float32)
        )

    def prepare_data(self, X_train, X_valid, X_test, y_train, y_valid, y_test):
        X_train = self.preprocessor.fit_transform(X_train)
        X_valid = self.preprocessor.transform(X_valid)
        X_test = self.preprocessor.transform(X_test)

        X_train_tensor, y_train_tensor = self._to_tensor(X_train, y_train)
        X_valid_tensor, y_valid_tensor = self._to_tensor(X_valid, y_valid)
        X_test_tensor, y_test_tensor = self._to_tensor(X_test, y_test)

        self.train_dataloader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size,
            shuffle=False
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




