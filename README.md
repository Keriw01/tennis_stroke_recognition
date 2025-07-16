# 🎾 Klasyfikacja Uderzeń w Tenisie

## 📌 Opis Projektu

Rozróżniamy 5 klas uderzeń:
- **forhend** (ang. forehand groundstroke, czyli uderzenie po odbiciu piłki od kortu)
- **bekhend** (ang. backhand groundstroke, czyli uderzenie po odbiciu od kortu)
- **forhend wolej** (ang. forehand volley)
- **bekhend wolej** (ang. backhand volley)
- **serwis/smecz** (ang. serve/smatch) (uderzenie od góry)

Tenisiści obecni na filmach, to:
- **Novak Djokovic**
- **Carlos Alcaraz**
- **Pablo Carreño-Busta**
- **Taylor Fritz**
- **Jack Sock**
- **Nieznany** sparingpartner Djokovica z drugiego filmu (z białą koszulką)

Pierwsze człony anotacji nazywają się odpowiednio:
1. **FH**
2. **BH**
3. **FHV**
4. **BHV**
5. **S**

Drugie człony anotacji mogą się nazywać odpowiednio:
1. **ND**
2. **CA**
3. **PBC**
4. **TF**
5. **JS**
6. **U** [od słowa unknown]

Przykład anotacji: Jeśli **Novak Djokovic** wykonuje uderzenie typu **forhend volley**, to anotacja będzie się nazywać **FHV ND**

### Główne elementy:
- Ekstrakcja współrzędnych punktów kluczowych ciała z użyciem **MediaPipe Pose**. Korzystamy z węzłów o następujących numerach: 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28
- Klasyfikacja za pomocą **k-Najbliższych Sąsiadów (k-NN)** z metryką **Dynamic Time Warping (DTW)** (`fastdtw`)

### Tryby normalizacji danych:
1. Uniezależnienie od położenia (punkt środkowy zawsze w punkcie zerowym)
2. Uniezależnienie od położenia (punkt w punkcie zerowym tylko w pierwszej klatce, a potem szkielet się porusza tak jak w oryginalnej akcji)
3. Uniezależnienie od położenia + rozmiaru (punkt środkowy zawsze w punkcie zerowym)
4. Uniezależnienie od położenia + rozmiaru (punkt w punkcie zerowym tylko w pierwszej klatce, a potem szkielet się porusza tak jak w oryginalnej akcji)

## 📁 Struktura Katalogów

Poniżej znajduje się opis głównych folderów i plików projektu:

- `tennis_stroke_recognition/` – główny katalog projektu
  - `data/` – dane wejściowe i wyjściowe
    - `raw_videos/` – oryginalne pliki wideo (.mp4)
    - `processed_videos_30fps/` – wideo przekonwertowane do 30 klatek na sekundę
    - `annotations_elan/` – adnotacje w formacie ELAN (.eaf)
    - `annotations_csv/` – adnotacje wyeksportowane do formatu CSV
    - `processed_features/` – pliki `.npz` z cechami szkieletu (raw + znormalizowane)
  - `src/`
    - `train_dtw_knn.py` – skrypt do treningu i ewaluacji klasyfikatora DTW + k-NN
  - `tools/` – skrypty pomocnicze
    - `fetch_30fps_video.py` – konwersja wideo do 30 FPS
    - `sequences.py` – ekstrakcja punktów szkieletu z wideo
    - `1_normalization.py` – normalizacja cech ruchu (Uniezależnienie od położenia, punkt środkowy zawsze w punkcie zerowym)
    - `matlab_scripts/` – skrypty do eksportu adnotacji z programu ELAN
      - `ELAN m-funkcje/` – folder z funkcjami pomocniczymi dla Matlaba
      - `extract_elan_annotations_to_csv.m` – eksport anotacji do CSV
  - `README.md` – dokumentacja projektu

## 🧪 Jak Uruchomić Projekt

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/Keriw01/tennis_stroke_recognition.git
cd tennis_stroke_recognition
```

### 2. Tworzenie środowiska wirtualnego i instalacja zależności

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

## 🔄 Pipeline Przetwarzania Danych

### Krok 1: Pobranie i konwersja filmów do 30 FPS
Uruchom skrypt pobierający oryginalne wideo oraz konwertujący wszystkie filmy do 30 klatek na sekundę:

```bash
python tools/fetch_30fps_video.py
```

### Krok 2: Wypakuj wymagany plik i uruchom skrypt w Matlab aby uzyskać adnotacje w formatcie .csv
Wypakuj plik `annotations_elan.rar` (ścieżka ma być **data/annotations_elan/**... pliki anotacji). Następnie uruchom skrypt `extract_elan_annotations_to_csv.m` w Matlab, aby uzyskać plik .csv potrzebny do uruchomienia skryptu `sequences.py`

### Krok 3: Ekstrakcja cech (szkieletów)
Po zakończeniu konwersji uruchom ekstrakcję cech:

```bash
python tools/sequences.py
```
Ten skrypt:
* Wczytuje filmy oraz adnotacje
* Dla każdej adnotacji wyznacza sekwencję punktów szkieletu
* Zapisuje dane do pliku raw_sequences.npz


### Krok 4: Normalizacja cech
```bash
python tools/1_normalization.py
```
Skrypt wczytuje surowe dane i normalizuje je poprzez:
* Uniezależnienie od położenia (przesunięcie środka ciężkości do zera)
* Zapis danych do pliku 1_normalized_sequences.npz

### Krok 5: Trening i ocena modelu
```bash
python src/train_dtw_knn.py
```
Skrypt:
* Wczytuje znormalizowane dane
* Dzieli je na zbiór treningowy i testowy
* Trenuje klasyfikator DTW + k-NN
* Zapisuje i wyświetla:
    * Dokładność (accuracy)
    * Raport klasyfikacji
    * Macierz pomyłek (confusion matrix)

Wyniki zostaną zapisane do nowego folderu w:
```bash
data/dtw_knn_results/<timestamp>/
```

## 🛠️ Wersje oprogramowania
* ELAN: 6.1
* Python: 3.12.10
* pip: 25.1.1
