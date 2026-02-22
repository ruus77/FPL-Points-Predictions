import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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