from pathlib import Path
import sys

try:
    BASE_DIR = Path(__file__).resolve().parent.parent 
    
except NameError:
    BASE_DIR = Path().resolve().parent

DATA_DIR = BASE_DIR / 'Data'
SCRIPTS_PATH = BASE_DIR / 'Scripts'

FPL_DATA_PATH = DATA_DIR / 'FPL_data.parquet'
TIDY_DATA_PATH = DATA_DIR / 'FPL_data_tidy.parquet'
UNTIDY_DATA_PATH = DATA_DIR / 'FPL_data_untidy.parquet'
PREDICTED_STATS_PATH = DATA_DIR / 'player_data.parquet'

if str(SCRIPTS_PATH) not in sys.path:
    sys.path.append(str(SCRIPTS_PATH))