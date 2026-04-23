from pathlib import Path
import sys

# 1. Dynamiczne ustalenie głównego folderu projektu
# Path(__file__) wskazuje na ten plik (config.py), .parent to folder, w którym on leży.
BASE_DIR = Path(__file__).resolve().parent

# 2. Definicja folderów (Zwróć uwagę na wielkość liter - 'Data' i 'Scripts')
DATA_DIR = BASE_DIR / 'Data'
SCRIPTS_PATH = BASE_DIR / 'Scripts'

# 3. Ścieżki do konkretnych plików danych
# Dopasowane do nazw z Twojego screena (Case-sensitive!)
FPL_DATA_PATH = DATA_DIR / 'FPL_data.parquet'
TIDY_DATA_PATH = DATA_DIR / 'FPL_data_tidy.parquet'
UNTIDY_DATA_PATH = DATA_DIR / 'FPL_data_untidy.parquet'

# Poniżej ścieżka do predykcji - na screenie widziałem player_data.parquet, 
# upewnij się, że to ten plik ma być używany jako PREDICTED_STATS_PATH
PREDICTED_STATS_PATH = DATA_DIR / 'player_data.parquet'

# 4. Automatyczne dodawanie folderu Scripts do ścieżek Pythona
# Dzięki temu importy typu 'from data_utils...' będą działać wszędzie.
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.append(str(SCRIPTS_PATH))
