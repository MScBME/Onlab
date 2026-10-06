# Úszóteljesítmény-elemző Rendszer (Önálló Laboratórium)

Ez a projekt egy MSc Önálló laboratórium keretében készült, teljes körű gépi látás (computer vision) alapú rendszer, amely az úszók nyomon követésére és teljesítménymutatóik (sebesség, pozíció) valós világban (méterben) történő elemzésére szolgál, YOLO-alapú objektumfelismerés és perspektíva-normalizálás alkalmazásával.

## Főbb Jellemzők
- **Perspektíva Normalizálás:** A kamera nézőpontjának transzformációja egy szabványosított, felülnézeti sávképpé (Bird's Eye View), amely elengedhetetlen a pontos, valós méretű metrikák számításához.
- **YOLO Detektálás:** Ultralytics YOLO modell alkalmazása a robusztus és gyors úszókövetés érdekében.
- **Sebességelemzés:** Fejlett jelfeldolgozási technikák az outlierek szűrésére és a sebességgörbék simítására.
- **Adatfeldolgozási Folyamat:** Integrált eszközök a videóképkockák kinyerésére, az adathalmaz építésére és a modell tanítására.

---

## Első Lépések

### 1. Telepítés
A projekt futtatásához Python 3.10 vagy újabb verzió szükséges.
```bash
pip install -r requirements.txt
```

### 2. Projekt Struktúra
- `data/raw/`: Bemeneti videók tárolási helye.
- `config/`: JSON alapú konfigurációs fájlok a képkocka-kinyeréshez és a futtatási beállításokhoz.
- `scripts/`: A főbb munkafolyamatok belépési pontjai (scriptek).
- `src/`: Alapvető üzleti logika és modulok (detektálás, követés, elemzés, videófeldolgozás).

---

## Munkafolyamat

### 1. Fázis: Adatelőkészítés
1.  **Videók és Sávok Definiálása:** A `data/videos.json` fájl frissítése a videók elérési útjával és az egyes sávokat határoló 4 pontos koordinátákkal.
2.  **Események Meghatározása:** Specifikus klipek (kezdő és végpontok) hozzáadása a `training/events.json` fájlhoz.
3.  **Képkockák Kinyerése:** A tanító képek kinyerése a definiált események alapján.
    ```bash
    python scripts/extract_training_frames.py
    ```

### 2. Fázis: Tanítás
1.  **Címkézés:** A `training/data/` mappába kinyert képkockák annotálása (pl. CVAT vagy Roboflow eszközökkel).
2.  **Sávok Előfeldolgozása:** A címkézett képkockák transzformálása szabványosított, felülnézeti sávképekké.
    ```bash
    python src/training/dataset_builder.py
    ```
3.  **Modell Tanítása:**
    ```bash
    python scripts/train_model.py
    ```

### 3. Fázis: Inferencia és Elemzés
A nyomon követés és sebességelemzés futtatása egy kiválasztott, a `data/clips.json` fájlban definiált klipen.
```bash
# Detektálás futtatása a config/run_config.json fájlban megadott active_clip_id alapján
python scripts/run_inference.py

# Vagy paraméterek felülírása parancssorból
python scripts/run_inference.py --clip d2du_woman_freestyle --lane 4
```

---

## Eszközök
- **Felhasználói Felület (UI Viewer):** Képi felület a videóadatok és annotációk felfedezéséhez.
  ```bash
  python scripts/ui_viewer.py
  ```
- **Manuális Követő (Tracker):** Hagyományos gépi látás algoritmus (CSRT) alkalmazása a manuális követéshez és verifikációhoz.
  ```bash
  python scripts/run_manual_tracker.py
  ```

---

## Konfiguráció
- **`config/run_config.json`**: Az alapértelmezett modell elérési utak, konfidencia küszöbértékek és aktív klipek beállítása.
- **`config/extraction_config.json`**: A tanításhoz mintavételezett képkockák paramétereinek szabályozása.
