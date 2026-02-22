# FPL Points Predictions

## Opis projektu
Projekt ma na celu przewidywanie punktów zdobywanych przez zawodników w grze Fantasy Premier League (FPL) z wykorzystaniem technik uczenia maszynowego. Poprzez analizę historycznych statystyk, formy zawodników oraz poziomu trudności nadchodzących spotkań, zbudowane modele mają na celu wsparcie procesu decyzyjnego przy budowie optymalnego składu menedżerskiego.

## Wykorzystane technologie
* Język programowania: Python 3.14
* Uczenie maszynowe i Deep Learning: Scikit-Learn, PyTorch
* Przetwarzanie i analiza danych: Pandas, NumPy
* Wizualizacja danych: Matplotlib, Seaborn
* Środowisko analityczne: Jupyter Notebook

## Struktura repozytorium
* `Data/` - Katalog przeznaczony na zbiory danych (surowe oraz przetworzone). Ze względu na rozmiar plików (CSV/Parquet), mogą one być ignorowane w systemie kontroli wersji.
* `Notebooks/` - Notatniki Jupyter wykorzystywane do eksperymentów, eksploracyjnej analizy danych (EDA) oraz inżynierii cech (Feature Engineering).
* `Scripts/` - Główny kod źródłowy projektu, obejmujący skrypty operacyjne oraz modele predykcyjne.
  * `model_utils/` - Autorski pakiet narzędziowy zorganizowany w sposób modułowy:
    * `data.py` - Logika związana z obsługą i podziałem zbiorów danych.
    * `preprocessing.py` - Potoki transformacji (Pipeline, ColumnTransformer) oraz klasa `FPLDataPipe` odpowiedzialna za przygotowanie danych dla modeli opartych na bibliotece PyTorch.
    * `selector.py` - Klasa `ModelSelector` przeznaczona do optymalizacji hiperparametrów, trenowania oraz ewaluacji algorytmów uczenia maszynowego.

## Instrukcja uruchomienia

1. Sklonuj repozytorium na swój komputer lokalny:
   ```bash
   git clone [https://github.com/ruus77/FPL-Points-Predictions.git](https://github.com/ruus77/FPL-Points-Predictions.git)
   cd FPL-Points-Predictions
