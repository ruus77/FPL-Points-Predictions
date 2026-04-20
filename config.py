from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'Data' / 'FPL_data.parquet'

SCRIPTS_PATH = BASE_DIR / 'Scripts'
if str(SCRIPTS_PATH) not in sys.path:from pathlib import Path
import sys

BASE_DIR = Path(r'C:\Users\russa\PycharmProjects\FPL_Points_Predicting\FPL-Points-Predictions')

DATA_DIR = BASE_DIR / 'Data'
SCRIPTS_PATH = BASE_DIR / 'Scripts'

DATA_PATH = DATA_DIR / 'FPL_data.parquet'
UNTIDY_DATA_PATH = DATA_DIR / 'FPL_data_untidy.parquet'

if str(SCRIPTS_PATH) not in sys.path:
    sys.path.append(str(SCRIPTS_PATH))
    sys.path.append(str(SCRIPTS_PATH))


TIDY_DATA_PATH = DATA_DIR / 'FPL_data_tidy.parquet'

FPL_DATA_PATH = DATA_DIR / 'FPL_data.parquet'
