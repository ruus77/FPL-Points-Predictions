# FPL Points Predictions

## Opis projektu
Projekt "FPL Points Predictions" dedykowany jest zaawansowanej analizie i prognozowaniu liczby punktów zdobywanych przez zawodników w systemie Fantasy Premier League (FPL). Wykorzystując techniki uczenia maszynowego, system przetwarza dane historyczne, kalendarz spotkań oraz kluczowe metryki piłkarskie w celu optymalizacji decyzji menedżerskich przed nadchodzącymi kolejkami (Gameweeks).

## Kluczowe cele analityczne
* **Predykcja punktowa**: Szacowanie oczekiwanej liczby punktów (xP) dla zawodników na podstawie modeli statystycznych.
* **Identyfikacja zawodników typu "Differential"**: Poszukiwanie graczy o niskim współczynniku popularności (niski procent posiadania), którzy generują wysoką liczbę punktów. Strategia ta pozwala na budowanie przewagi nad szeroką populacją graczy poprzez wybór efektywnych, lecz niedocenianych zawodników.
* **Analiza efektywności kosztowej**: Badanie korelacji między ceną rynkową zawodnika a jego realnym potencjałem punktowym.

## Analiza rozkładu punktów względem popularności
W procesie analitycznym kluczowe znaczenie ma zrozumienie, jak popularność zawodnika (procent posiadania) koreluje z jego wynikami. Poniższa wizualizacja pozwala na szybkie zidentyfikowanie graczy o niskim posiadaniu, którzy osiągają wyniki powyżej średniej rynkowej.

> *Wykres przedstawia zależność punktową dla poszczególnych pozycji (GKP, DEF, MID, FWD). Analiza skupia się na identyfikacji outlierów w lewym górnym kwadrancie każdego podwykresu.*
>
> <img width="875" height="784" alt="Rozkład punktów względem popularności" src="https://github.com/user-attachments/assets/6bdf8a0b-905d-4411-bbee-93808ce3848e" />

## Analiza Statystyk Oczekiwanych (Wykresy Radarowe)
Projekt pozwala na bezpośrednie, wizualne porównanie profilu statystycznego zawodników za pomocą wykresów radarowych. Narzędzie to ułatwia zestawienie oczekiwanych metryk w kontekście konkretnej kolejki (np. GW 29).

**Porównanie graczy ofensywnych:**
<br>
<img width="950" height="850" alt="Statystyki Oczekiwane - Ofensywa" src="https://github.com/user-attachments/assets/31bcd4fe-e3e5-4996-9157-9a60fbe54996" />
<br>
*Zestawienie: Erling Haaland, Antoine Semenyo, Mohamed Salah, Bruno Fernandes.*

**Porównanie graczy defensywnych:**
<br>
<img width="950" height="850" alt="Statystyki Oczekiwane - Defensywa" src="https://github.com/user-attachments/assets/da30cff3-ca9f-4f2a-9da3-411b28a90917" />
<br>
*Zestawienie: Kenny Tete, Marc Guéhi, Lewis Hall, Trevoh Chalobah.*

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
