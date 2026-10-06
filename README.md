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
A projekt futtatásához Python 3.10 vagy újabb verzió szükséges. A függőségeket a projekt virtuális környezetébe (`.onlab_venv`) telepítsük, ne a globális Pythonba:
```bash
.onlab_venv/Scripts/python.exe -m pip install -r requirements.txt
```
A tesztek futtatása:
```bash
.onlab_venv/Scripts/python.exe -m pytest
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
- **Asztali alkalmazás (Swimmer Tracker):** PySide6 + Qt Quick felület (a felület szövegei angolok).
  ```bash
  .onlab_venv/Scripts/python.exe scripts/run_app.py
  ```
  - *Video library:* a `data/videos.json` videói kártyákon. Az **Add video** egy fájlt a `data/raw/` mappába másol (ha már ott van, helyben regisztrálja), és legalább egy lane megadását kéri.
  - *Lane editor:* a lane 4 sarka bármelyik képkockán, tetszőleges sorrendben kattintható, a mentett sorrend automatikusan normalizálódik. Lane-enként megadható a kijelölt szakasz valós hossza (`length_m`, alapértelmezés 25 m). Lane-ek másolhatók egy másik videóból. Klip vagy tanítóadat által használt lane nem törölhető.
  - *Workspace:* időtartomány kijelölése a timeline-on (I/O billentyű, „Zoom to range”), majd lane, modell és konfidencia választása. Az elemzés élőben mutatja a detekciót, a bird's-eye nézetet, a telemetriát és a sebességgrafikont, a végén pedig összegzést ad. Elérhető a Pause/Resume, a Restart és a Stop. A beállítások változása csak Restart után érvényesül.
  - *Create clip…:* az aktuális tartományt klipként menti a `data/clips.json`-be. Az id automatikus (`<video_id>_lane<n>_<HH>_<MM>_<SS>`), csak a leírást kell megadni.
- **Manuális Követő (Tracker):** Hagyományos gépi látás algoritmus (CSRT) alkalmazása a manuális követéshez és verifikációhoz.
  ```bash
  python scripts/run_manual_tracker.py
  ```

---

## Konfiguráció
- **`config/run_config.json`**: Az alapértelmezett modell elérési utak, konfidencia küszöbértékek és aktív klipek beállítása.
- **`config/extraction_config.json`**: A tanításhoz mintavételezett képkockák paramétereinek szabályozása.
