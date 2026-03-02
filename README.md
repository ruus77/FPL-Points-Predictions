# FPL Points Predictions

## Opis projektu
Projekt "FPL Points Predictions" dedykowany jest zaawansowanej analizie i prognozowaniu liczby punktów zdobywanych przez zawodników w systemie Fantasy Premier League (FPL). Wykorzystując techniki uczenia maszynowego, system przetwarza dane historyczne, kalendarz spotkań oraz kluczowe metryki piłkarskie w celu optymalizacji decyzji menedżerskich przed nadchodzącymi kolejkami (Gameweeks).

## Kluczowe cele analityczne
* **Predykcja punktowa**: Szacowanie oczekiwanej liczby punktów (xP) dla zawodników na podstawie modeli statystycznych.
* **Identyfikacja zawodników typu "Differential"**: Poszukiwanie graczy o niskim współczynniku popularności (niski procent posiadania), którzy generują wysoką liczbę punktów. Strategia ta pozwala na budowanie przewagi nad szeroką populacją graczy poprzez wybór efektywnych, lecz niedocenianych zawodników.
* **Analiza efektywności kosztowej**: Badanie korelacji między ceną rynkową zawodnika a jego realnym potencjałem punktowym.

## Analiza rozkładu punktów względem popularności
W procesie analitycznym kluczowe znaczenie ma zrozumienie, jak popularność zawodnika (procent posiadania) koreluje z jego wynikami. Poniższa wizualizacja pozwala na szybkie zidentyfikowanie graczy o niskim posiadaniu, którzy osiągają wyniki powyżej średniej rynkowej.
>
> *Wykres przedstawia zależność punktową dla poszczególnych pozycji (GKP, DEF, MID, FWD). Analiza skupia się na identyfikacji outlierów w lewym górnym kwadrancie każdego podwykresu*.<img width="872" height="790" alt="popularity_vs_points" src="https://github.com/user-attachments/assets/96e75bef-73ef-4a5f-9536-a338bace6293" />




## Struktura repozytorium
* **Data/**: Przechowywanie zbiorów danych wejściowych oraz wyników przetworzonych (duże pliki są wyłączone z kontroli wersji).
* **Notebooks/**: Dokumentacja procesu eksploracyjnej analizy danych (EDA), prototypowanie inżynierii cech oraz walidacja modeli.
* **Scripts/**: Produkcyjne skrypty Python realizujące procesy ETL oraz algorytmy predykcyjne.

## Technologie i wymagania
* **Język**: Python 3.14
* **Analiza danych**: Pandas, NumPy
* **Wizualizacja**: Matplotlib, Seaborn
* **Modelowanie**: Scikit-learn

## Instrukcja uruchomienia lokalnego
W celu poprawnej konfiguracji środowiska należy wykonać poniższe kroki:

1. Klonowanie repozytorium:
   ```bash
   git clone [https://github.com/ruus77/FPL-Points-Predictions.git](https://github.com/ruus77/FPL-Points-Predictions.git)
   cd FPL-Points-Predictions
