import pandas as pd
pd.set_option('display.max_columns', None)
from warnings import filterwarnings
filterwarnings("ignore")

import config
from data_utils import data_import, sort_data, id_cols_detection, duplicated_cols


data = data_import()

data = id_cols_detection(data)

data = sort_data(data)
data = data.convert_dtypes()


dup_cols = duplicated_cols(df=data)
data.drop(dup_cols, axis=1, inplace=True)

data.to_parquet(config.UNTIDY_DATA_PATH, index=False)
