# FPL Points Predictions

## Opis projektu
Projekt "FPL Points Predictions" ma na celu prognozowanie liczby punktów zdobywanych przez zawodników w grze Fantasy Premier League (FPL). Rozwiązanie opiera się na technikach uczenia maszynowego, analizując dane historyczne, zaawansowane metryki piłkarskie oraz kalendarz spotkań. Projekt stanowi narzędzie analityczne wspomagające podejmowanie decyzji przy optymalizacji składu w nadchodzących kolejkach (Gameweeks).

## Struktura repozytorium
Poniższe zestawienie prezentuje podział logiczny plików i katalogów wewnątrz projektu:

* `Data/` - Katalog przeznaczony na zbiory danych wejściowych i wyjściowych. Ze względu na politykę zarządzania przestrzenią w repozytorium, duże pliki danych są ignorowane przez system kontroli wersji (konfiguracja w pliku `.gitignore`).
* `Notebooks/` - Zbiór notatników Jupyter (format `.ipynb`) wykorzystywanych do eksploracyjnej analizy danych (EDA), testowania hipotez, prototypowania procesów inżynierii cech oraz wstępnej oceny modeli.
* `Scripts/` - Główny zbiór skryptów napisanych w języku Python. Znajdują się tu moduły odpowiedzialne za przetwarzanie potokowe (pipeline), czyszczenie i transformację zbiorów danych, a także algorytmy implementujące wyuczone modele predykcyjne.

## Technologie i wymagania
Projekt został zrealizowany w oparciu o następujący stos technologiczny:
* **Język:** Python 3.14
* **Analiza i transformacja danych:** Pandas, NumPy
* **Wizualizacja wyników:** Matplotlib, Seaborn
* **Środowisko deweloperskie:** Jupyter Notebook

## Uruchomienie lokalne
W celu odtworzenia środowiska i uruchomienia projektu na lokalnej stacji roboczej, zalecane jest korzystanie z izolowanego środowiska wirtualnego. Należy wykonać następujące kroki:

1. Klonowanie repozytorium na dysk lokalny:
   ```bash
   git clone [https://github.com/ruus77/FPL-Points-Predictions.git](https://github.com/ruus77/FPL-Points-Predictions.git)
   cd FPL-Points-Predictions
