import pandas as pd
pd.set_option('display.max_columns', None)
from warnings import filterwarnings
filterwarnings("ignore")

import config
from data_utils import data_import, sort_data

data = data_import()
data = sort_data(data)

data = data.convert_dtypes()


id_cols = [c for c in data.columns
    if (
        ("_id" in c or "id_" in c or c == "id")
        or "code" in c
        or "name" in c
        or "position" in c
        or 'status' in c
        or data[c].dtype == 'object'
        or c == "gw"
    )
    or c in ["season_id", "gw"]]

data = data[id_cols + [c for c in data.columns if c not in id_cols]]


def duplicated_cols(df:pd.DataFrame)->list[str]:
    return df.columns[df.T.duplicated(keep=False)].tolist()

dup_cols = duplicated_cols(df=data)
data.drop(dup_cols, axis=1, inplace=True)

data.to_parquet(config.UNTIDY_DATA_PATH, index=False)