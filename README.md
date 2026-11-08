# Úszóteljesítmény-elemző Rendszer (Önálló labor)

Ez a projekt egy MSc Önálló laboratórium keretében készült, teljes körű gépi látás alapú rendszer, amely az úszók nyomon követésére és teljesítménymutatóik valós világban történő elemzésére szolgál, YOLO-alapú objektumfelismerés és perspektíva-normalizálás alkalmazásával. A rendszer rögzített kamerával készült medencefelvételeken dolgozik. A felhasználó a videón négy sarokponttal kijelöli a sávokat, a rendszer pedig a kiválasztott sávot felülnézeti képpé alakítja, amelyen a sáv hossza a kép függőleges tengelye mentén egyenletes léptékű, így az úszó helyzete közvetlenül méterben mérhető. Ezen a képen egy saját annotált képkockákon tanított YOLO modell keresi meg az úszót, a sebességet pedig egy Kalman-szűrő becsli, amely a víz alatti, észleletlen szakaszokat is áthidalja. Elemzés közben élőben látszik a detekció, a pozíció és a sebesség, a végén pedig simított sebességprofil, átlag- és maximális sebesség készül. A projekt egy Windowsra telepíthető asztali alkalmazásból és a mögötte álló kutatási kódból áll. Az alkalmazás a betanított modellek ONNX-exportját processzoron futtatja, így nem kell hozzá videókártya, a kutatási kód pedig a tanítóadat előkészítését, a modellek tanítását és kiértékelését végzi.

## Telepítés

Az alkalmazás 64 bites Windows 10 vagy 11 rendszerre a `SwimmerTracker-<verzió>-setup.exe` telepítővel tehető fel. A telepítéshez nem kell rendszergazdai jog, és Pythont vagy más programot sem kell előre telepíteni.

1. Indítsd el a telepítőt. A telepítő nincs digitálisan aláírva, ezért a Windows SmartScreen figyelmeztetést mutathat. Ilyenkor a **További információ**, majd a **Futtatás mindenképp** gombbal lehet továbblépni.
2. A telepítő angol nyelvű. A program a saját felhasználói fiókod alá települ, alapértelmezésben a `%LOCALAPPDATA%\Programs\Swimmer Tracker` mappába. A telepítő asztali parancsikon készítését is felajánlja.
3. A telepítés végén az alkalmazás rögtön elindítható, később a Start menüből vagy az asztali parancsikonnal indul.

Új verzióra frissítéshez elég az új telepítőt lefuttatni, a régi verzió helyére települ. Eltávolítani a Windows Beállítások alkalmazásban, a telepített alkalmazások listájában lehet. Sem a frissítés, sem az eltávolítás nem törli a videókönyvtárat és a beállításokat, ezek helyét a Felhasználói útmutató leírja.

## Felhasználói útmutató

Az alkalmazás felülete angol nyelvű, ezért az útmutató a gombokat és a mezőket a felületen látható nevükkel említi.

### A videókönyvtár

Indításkor a **Video library** oldal jelenik meg, amely a könyvtár minden videóját egy-egy kártyán mutatja. A kártyán látszik egy bélyegkép, a videó azonosítója, fájlneve, hossza, felbontása és képkockasebessége, valamint a sávjai és a klipjei száma. A kártyára kattintva a videó megnyílik az elemző nézetben, az **Edit lanes** gombbal pedig a sávjai szerkeszthetők. Ha egy videó fájlja hiányzik, a kártya **File missing** jelzést kap, és nem nyitható meg. A **Refresh** gomb újraolvassa a könyvtárat.

### Videó hozzáadása

Az **Add video** gombbal választható ki egy videófájl, a támogatott formátumok az mp4, mov, avi, mkv és m4v. A megnyíló ablak tetején a fájl adatai látszanak, alatta a sávszerkesztő.

- Az alkalmazás a fájlt a háttérben bemásolja a könyvtár `videos` mappájába. Ha a fájl már ott van, nem másolja, csak felveszi a könyvtárba. A másolás alatt már ki lehet jelölni a sávokat.
- A **Video ID** a videó egyedi azonosítója. A fájlnévből automatikusan kitöltődik, és csak betűt, számjegyet és aláhúzásjelet tartalmazhat.
- A **Save video** gomb akkor használható, ha a **Video ID** érvényes, legalább egy sáv ki van jelölve, és egyik sáv sem hibás. Ha a másolás még tart, a mentés a másolás végén magától megtörténik.
- A **Cancel** megszakítja a hozzáadást, és törli a már bemásolt fájlt.

### Sávok kijelölése

A sávszerkesztő a videó hozzáadásakor, illetve később az **Edit lanes** gombbal nyílik meg. Bal oldalon a videó egy képkockája látszik a sávokkal, jobb oldalon a sávok listája és a kiválasztott sáv felülnézeti előnézete.

1. A **Frame** csúszkával válassz egy képkockát, amelyen a sáv jól látszik.
2. Kattints az **Add lane** gombra, majd a sáv négy sarkára. A sorrend tetszőleges, az alkalmazás automatikusan rendezi a pontokat. Az Esc billentyű megszakítja a kijelölést.
3. A kiválasztott sáv sarkai húzással finomíthatók. A sávlistában egy sorra kattintva másik sáv választható ki.
4. A **Bird's-eye** előnézet mutatja, hogyan látja a sávot az elemzés. Jó kijelölésnél a sávot határoló kötelek az előnézet két szélén egyenes, függőleges vonalként látszanak.
5. Az új sáv sorszáma a listában átírható, a már elmentett sávoké nem. A **Length** mezőbe a kijelölt szakasz valós hossza kerül méterben, ez alapértelmezésben 25. Ha a kijelölés nem a teljes medencehosszt fedi le, itt a ténylegesen kijelölt rész hosszát kell megadni, különben a pozíció és a sebesség hibás lesz.
6. A **Save lanes** gomb menti a sávokat. Új videónál ezt a **Save video** gomb teszi meg.

A kiválasztott sáv két végén a **0 m** és a sáv hosszát mutató felirat látszik, az alkalmazás a pozíciót a 0 m-es végtől méri. A hibás sávot a lista **invalid** felirattal jelöli, a hiba oka a lista alatt olvasható. Amelyik sávot egy elmentett klip használja, az nem törölhető, ezt az **in use** felirat jelzi. A **Copy lanes from** listában kiválasztott másik videó sávjai a **Copy** gombbal másolhatók át, ami az azonos kameraállásból készült felvételeknél hasznos. A másolás a jelenlegi sávokat lecseréli.

### Elemzés

Az elemző nézetben a videó látszik a sávokkal, mellette a kiválasztott sáv felülnézeti képe, alatta az idővonal és a sebességgrafikon, jobb oldalon pedig az elemzés beállításai és eredményei. A bal felső **Library** gomb visz vissza a könyvtárhoz.

**Lejátszás és tekerés.** Az idővonalon kattintással vagy húzással lehet tekerni. A lejátszás gomb vagy a Szóköz detekció nélküli előnézetet játszik le. A gombsor többi gombja egy másodpercet vagy egy képkockát lép előre vagy vissza.

**Az elemzett szakasz.** Az idővonal kiemelt része az elemzendő szakasz. A két vége húzható, a **Set start** és a **Set end** gombbal vagy az I és O billentyűvel az aktuális pozícióra állítható, illetve a **Start** és az **End** mezőbe beírható `HH:MM:SS.mmm` alakban. Hosszú videónál a **Zoom to range** gomb az idővonalat a kijelölt szakaszra nagyítja.

**Beállítások.** Az **Analysis** panelen kell kiválasztani a sávot, a modellt és a **Confidence** küszöböt. A küszöb az a legkisebb biztosság, amely mellett a modell találatát úszónak fogadjuk el. Magasabb küszöbnél kevesebb a téves találat, de több képkockán marad el a detekció. Az alkalmazás megjegyzi az utoljára használt modellt és küszöböt.

**Futtatás.** Az **Analyze** gomb elindítja az elemzést a kijelölt szakaszon. A modell első betöltése néhány másodpercig tarthat. Az elemzés minden képkockát feldolgoz, és ha a gép bírja, a videó valós sebességével halad. Futás közben ugyanez a gomb szünetelteti, majd folytatja az elemzést, a **Stop** leállítja, a **Restart** elölről kezdi. Futás közben a tekerés nem lehetséges. Ha a sávot, a modellt, a küszöböt vagy a szakaszt menet közben vagy a futás után módosítod, az új beállítás csak az elemzés újraindításakor érvényes, erre a panelen egy figyelmeztetés is utal.

**Élő adatok.** A **Live telemetry** panel az aktuális sebességet, a pozíciót méterben, a detekció biztosságát, a feldolgozás sebességét képkocka per másodpercben és az eltelt időt mutatja. A videón egy pont jelzi az úszó helyét, a felülnézeti képen pedig zöld keret a detekciót. A **Status** mező értékei a következők.

| Állapot | Jelentés |
|---|---|
| **Detected** | A modell ezen a képkockán megtalálta az úszót. A pont zöld. |
| **Kalman prediction** | Ezen a képkockán nincs detekció, például mert az úszó a víz alatt van, ezért a pozíciót a Kalman-szűrő becsli. A pont narancssárga. |
| **Searching** | Az elemzés kezdete óta még egyetlen detekció sem volt. |

**Sebességgrafikon és összegzés.** A grafikon futás közben a Kalman-szűrő élő sebességbecslését rajzolja, a futás végén pedig megjelenik a simított sebességprofil is. A **Summary** panel az átlag- és a maximális sebességet, a detekciós arányt és az elemzett időt mutatja, és jelzi, ha az elemzés a szakasz vége előtt leállt. A simítás a 3,2 m/s fölötti, fizikailag valószínűtlen értékeket levágja, ezért a 3,2 m/s körüli maximális sebesség instabil detekcióra utal. Ha túl kevés képkockán volt detekció, sebességstatisztika nem készül.

### Klipek

A klip egy elmentett szakasz egy sávval, amely később egy kattintással visszatölthető. A **Create clip** gomb a kijelölt szakaszt és sávot klipként menti. Az azonosítót az alkalmazás adja, csak egy leírás adható meg hozzá. Ugyanaz a szakasz ugyanazzal a sávval nem menthető el kétszer. A **Saved clips** panelen egy klipre kattintva betöltődik a sávja és a szakasza.

### Billentyűparancsok

Az elemző nézetben a következő billentyűk használhatók.

| Billentyű | Művelet |
|---|---|
| Szóköz | Az előnézet lejátszása vagy megállítása |
| Bal és jobb nyíl | Egy másodperc vissza vagy előre |
| Shift és bal vagy jobb nyíl | Egy képkocka vissza vagy előre |
| I | A szakasz eleje az aktuális pozícióra |
| O | A szakasz vége az aktuális pozícióra |

A sávszerkesztőben az Esc megszakítja az új sáv kijelölését.

### Fájlok és beállítások

A videókönyvtár a `Dokumentumok\Swimmer Tracker` mappa. Ebben a `videos` mappa tartalmazza a videófájlokat, a `videos.json` a videók és sávjaik adatait, a `clips.json` pedig a klipeket. A könyvtár egyben menthető vagy másik gépre vihető. Videót és klipet a felületről egyelőre nem lehet törölni. Az utoljára használt modellt és küszöböt az alkalmazás a `%APPDATA%\Swimmer Tracker\Swimmer Tracker.ini` fájlban tárolja. A modellek az alkalmazással együtt települnek és frissülnek.

## Fejlesztőknek

### A kódbázis struktúrája

| Mappa | Tartalom |
|---|---|
| `swimmertracker/` | Az asztali alkalmazás, ez kerül a telepítőbe. A mag a `video`, `detection`, `tracking`, `analysis`, `catalog` és `playback` csomag, a Qt Quick felület az `ui` csomagban van. |
| `research/` | A kutatási kód, amely nem kerül a telepítőbe. A `training/` a tanítóadat előkészítése és a tanítás, az `evaluation/` a kiértékelő parancsok, a `csrt/` a korábbi, kézi CSRT-követéses kísérlet. |
| `research/training/library/` | A train videókönyvtár. |
| `research/training/metadata/` | A tanítóadat leírása, vagyis az `events.json`, az `event_frames.json`, a `custom_frames.json` és az `extraction_config.json`. |
| `research/training/cache/` | A tanítás letöltött és köztes fájljai, ez nincs a gitben. |
| `data/` | A test videókönyvtár, egyben az alkalmazás fejlesztői könyvtára. |
| `models/` | A modellek, mindegyikhez egy `.pt`, egy `.onnx` és egy `.json` metaadat-fájl tartozik. Ez a mappa a kutatás és az alkalmazás közti átadási pont, és nincs a gitben. |
| `packaging/` | A telepítő buildje, vagyis a PyInstaller spec, az Inno Setup script, a kiadott modellek listája és a build script. |
| `tests/` | A pytest tesztek. |

Az alkalmazás soha nem importálja a `research` csomagot, a torch-ot, az ultralytics-et és a tanítás többi csomagját, így ezek nem kerülnek a telepítőbe. A felület nem használ OpenCV-t és onnxruntime-ot, a mag pedig Qt-t. Ezeket a szabályokat a `tests/test_layering.py` ellenőrzi.

#### A train és a test videókönyvtár

Mindkét könyvtár ugyanolyan szerkezetű, mint a felhasználói könyvtár, vagyis egy `videos.json`, egy `clips.json` és egy `videos/` mappa a videófájlokkal. A train könyvtár videóiból kinyert képkockák csak a train és a val halmazba kerülnek, a test könyvtáréiból kinyertek csak a test halmazba. A modelleket az alkalmazásban a test könyvtár videóin próbáljuk ki. Egy videóazonosító és egy videófájl csak az egyik könyvtárban szerepelhet, ezt a tanító kód betöltéskor ellenőrzi. Ha egy test videóból tanítóadat lesz, a fájlt a `videos.json` bejegyzésével és a klipjeivel együtt át kell tenni a train könyvtárba.

### Setup

A fejlesztői környezet Windowson van kipróbálva. A Python verziót és a függőségeket az [uv](https://docs.astral.sh/uv/) kezeli, a megfelelő Pythont is ő tölti le, ezért előre csak a Gitre és az uv-re van szükség. A tanításhoz CUDA-képes NVIDIA videókártya ajánlott, a telepítő buildeléséhez pedig az Inno Setup kell, erről a Telepítő buildelése rész szól.

1. Telepítsd az uv-t, majd nyiss új terminált, hogy az `uv` parancs elérhető legyen.
   ```powershell
   winget install --id=astral-sh.uv -e
   ```
2. Klónozd a repót, és hozd létre a virtuális környezetet.
   ```bash
   git clone https://github.com/MScBME/Onlab.git
   cd Onlab
   uv sync
   ```
   Az `uv sync` a projekt mappájában létrehozza a `.venv` környezetet a `.python-version` szerinti Pythonnal, és telepíti az alkalmazás, a kutatási kód és a tesztek függőségeit. A torch a CUDA 12.6-os PyTorch-indexről települ. A `pyproject.toml` vagy az `uv.lock` változása után a parancsot újra le kell futtatni.
3. Másold be a videókat és a modelleket, mert ezek nincsenek a gitben. A videók a `data/videos/` és a `research/training/library/videos/` mappába kerülnek, a `videos.json` fájlokban szereplő néven, a modellek pedig a `models/` mappába.
4. Ellenőrizd a környezetet a tesztekkel.
   ```bash
   uv run pytest
   ```

Minden parancsot `uv run <parancs>` alakban kell futtatni, így a projekt környezetében fut, és semmit nem kell aktiválni. Csomagot ne telepíts pip-pel, mert minden függőség a `pyproject.toml` fájlban van. Új függőséget az `uv add <csomag>` paranccsal lehet felvenni, a csak a kutatáshoz kellőt az `uv add --group research <csomag>` paranccsal.

### Munkafolyamat

A modellek fejlesztése három fázisból áll, a kiadást a telepítő buildelése zárja. A parancsok listája a `pyproject.toml` `[project.scripts]` szakaszában van, és mindegyik kiírja a saját súgóját a `--help` kapcsolóval. Az alábbi leírásokban a `<...>` egy megadandó értéket jelöl, a `[...]` pedig elhagyható részt.

#### Az alkalmazás futtatása forrásból

```bash
uv run swimmertracker [--library <dir>]
```

| Kapcsoló | Jelentés |
|---|---|
| `--library <dir>` | A megnyitandó videókönyvtár mappája. Forrásból futtatva alapértelmezésben a repó `data/` mappája, telepítve a `Dokumentumok\Swimmer Tracker`. Ha a mappa nem létezik, az alkalmazás létrehozza. |

Két környezeti változó is befolyásolja az alkalmazást.

| Változó | Jelentés |
|---|---|
| `SWIMMERTRACKER_LIBRARY=<dir>` | A videókönyvtár mappája, ha nincs megadva `--library` kapcsoló. |
| `SWIMMERTRACKER_MODELS=<dir>` | A modellek mappája. Forrásból futtatva alapértelmezésben a `models/`, telepítve az alkalmazással szállított modellek mappája. |

Az alkalmazás a modellek mappájának minden `.onnx` modelljét felkínálja, kivéve azokat, amelyeknek a metaadat-fájlja nem a sávos, `lane_warp` típusú bemenetet írja elő. A modellválasztóban a metaadat-fájl `display_name` mezője látszik.

#### 1. Adatelőkészítés

**Videók és sávok.** A tanítóvideókat és a sávjaikat az alkalmazásban kell felvenni úgy, hogy a train könyvtárat nyitjuk meg vele.

```bash
uv run swimmertracker --library research/training/library
```

**Események.** A `research/training/metadata/events.json` sorolja fel azokat a videószakaszokat, amelyekből képkockát kell kinyerni. Egy esemény mezői a `video_id`, a `HH:MM:SS` alakú `start_time` és `end_time`, a `stroke`, vagyis az úszásnem, a `gender`, valamint a `lanes`, azoknak a sávoknak a listája, amelyekben úszó van. Az esemény bármelyik könyvtár videójára vonatkozhat, a képkocka szerepét a videó könyvtára dönti el.

**Képkockák kinyerése.**

```bash
uv run train-extract [kapcsolók]
```

A parancs törli a `research/training/cache/raw_frames/` mappát, és az események alapján újra feltölti. A már regisztrált képkockákat kihagyja, mert azok feltehetően már fent vannak a Roboflow-n, az újakat pedig felveszi az `event_frames.json` fájlba. Egy esemény módosítása új képkocka-azonosítókat hoz létre, a régi bejegyzések megmaradnak. Képkockát törölni az `event_frames.json` bejegyzésének és a Roboflow-n lévő képnek a kézi törlésével lehet.

| Kapcsoló | Jelentés |
|---|---|
| `--filter <key>=<value> ...` | Csak a feltételeknek megfelelő események. Több feltételnek egyszerre kell teljesülnie. A kulcs az esemény bármelyik mezője lehet, a `lanes=<n>` azokat az eseményeket választja ki, amelyekben az adott sáv szerepel. |
| `--event-ids <n>,<n>,...` | Csak az `events.json` megadott sorszámú eseményei, egytől számozva. A `--filter` ezeken belül szűr. |
| `--frame-count [<n>]` | Eseményenként pontosan `<n>` képkocka az esemény elejétől, a mintavételi időközzel. Érték nélkül az `extraction_config.json` `frame_count` értéke érvényes. Ha az utolsó képkocka az esemény vége után esne, a parancs hibával leáll. |
| `--interval-sec <seconds>` | A mintavételi időköz másodpercben az `extraction_config.json` `default_interval_sec` értéke helyett. |

A kinyerés alapbeállításait a `research/training/metadata/extraction_config.json` tartalmazza. A `default_interval_sec` a mintavételi időköz másodpercben, a `frame_count` pedig az eseményenkénti képkockaszám, amely csak a `--frame-count` kapcsolóval érvényes.

**Annotálás.** A kinyert képkockákat a `research/training/cache/raw_frames/` mappából kell feltölteni a Roboflow-ra, és ott annotálni. Az annotáció az úszó fejét és felsőtestét fogja körbe. Ha a Roboflow-projektnek több osztálya van, a tanítás mindet egyetlen `swimmer` osztályba vonja össze.

#### 2. Tanítás

A tanítás a Roboflow-ról tölti le az annotált képkockákat. Ehhez a projekt gyökerében egy `.env` fájl kell a következő tartalommal.

```
ROBOFLOW_API_KEY=<api_key>
ROBOFLOW_WORKSPACE=<workspace>
ROBOFLOW_PROJECT=<project>
ROBOFLOW_VERSION=<version>
```

A letöltött annotációk a `research/training/cache/annotated/` mappába kerülnek. A parancs csak akkor használja őket újra, ha ugyanabból a projektből és verzióból származnak, egyébként újra letölti őket.

```bash
uv run train-model --name <name> [kapcsolók]
```

A parancs egy lépésben letölti az annotált képkockákat, kiválasztja a szűrőknek megfelelőket, és mindegyiket sávonként felülnézeti képpé alakítja a címkéivel együtt. Ezután train, val és test halmazra osztja az adatot, betanítja a YOLO modellt, kiértékeli a val és a test halmazon, végül ONNX formátumba exportálja az alkalmazás számára. Az eredmény a `models/swimmer_<name>.pt`, a `models/swimmer_<name>.onnx` és a `models/swimmer_<name>.json` metaadat-fájl. A futás részletei, például a tanulási görbék, a `research/training/cache/runs/<name>/` mappába kerülnek.

A test halmaz a test könyvtár videóinak minden annotált képkockája. A train könyvtár képkockái csoportonként kerülnek a train vagy a val halmazba. Egy csoportot ugyanannak a videófájlnak az egymáshoz `--split-gap-s` másodpercnél közelebbi képkockái alkotnak, így a szinte azonos szomszédos képkockák nem kerülhetnek két különböző halmazba. A csoportokat úszásnemenként úgy osztja el, hogy minden úszásnem képeinek nagyjából `--val-fraction` része kerüljön a val halmazba, és minden úszásnemből maradjon legalább egy csoport a train halmazban. Ha még nincs annotált képkocka a test könyvtár videóiból, a test kiértékelés kimarad.

| Kapcsoló | Jelentés |
|---|---|
| `--name <name>` | Kötelező. A modell neve, a kimeneti fájlok neve `swimmer_<name>`. Létező név esetén a korábbi modell felülíródik, de a kézzel beállított megjelenített neve megmarad. |
| `--filter <key>=<value> ...` | Csak a feltételeknek megfelelő képkockák. A kulcs `video_id`, `stroke` vagy `gender` lehet, több feltételnek egyszerre kell teljesülnie. |
| `--frames <pattern>,<pattern>,...` | Csak azok a képkockák, amelyek fájlneve illeszkedik valamelyik glob mintára, például `<video_id>_*.jpg`. Az egyedi képkockák neve `custom_<n>.jpg`. |
| `--epochs <n>` | A tanítási epochok száma, alapértelmezésben 50. |
| `--imgsz <px>` | A tanítási képméret pixelben, alapértelmezésben 1024. |
| `--batch <n>` | A batch mérete, alapértelmezésben 16. |
| `--seed <n>` | A tanítás véletlenmagja, alapértelmezésben 0. Több maggal futtatva mérhető az eredmények szórása. |
| `--rect`, `--no-rect` | Az alapértelmezett `--rect` téglalap alakú batchekkel, mosaic augmentáció nélkül tanít, a `--no-rect` négyzetes képekkel és mosaic augmentációval. |
| `--base-weights <file>` | A kiinduló súlyok, alapértelmezésben `yolo11n.pt`. A puszta fájlnevet a `research/training/cache/weights/` mappában keresi, és ha nincs ott, az ultralytics oda tölti le. |
| `--val-fraction <fraction>` | Az úszásnemenkénti képek val halmazba kerülő aránya, alapértelmezésben `0.15`. |
| `--split-gap-s <seconds>` | Az ennél közelebbi képkockák ugyanabba a csoportba kerülnek, alapértelmezésben 60 másodperc. |
| `--split-seed <n>` | A felosztás véletlenmagja, alapértelmezésben 42. |
| `--force-download` | Akkor is újra letölti az annotációkat, ha a meglévők naprakészek. |
| `--skip-download` | Nem tölt le semmit, a meglévő annotációkat használja. |
| `--skip-train` | A felosztás után leáll, így a halmazok összetétele tanítás nélkül is ellenőrizhető. |

A `models/<name>.json` metaadat-fájl kapcsolja össze a kutatást az alkalmazással. A `display_name` a modellválasztóban látszó név, ez kézzel átírható, és a tanítás sosem írja felül. Az `input` a modell bemenetének típusa és mérete. A `training` rögzíti, milyen git commitból, milyen Roboflow-adatból, milyen szűrőkkel, felosztással és paraméterekkel készült a modell, a `metrics` a val és a test metrikákat, az `onnx` pedig az export adatait tartalmazza.

#### 3. Kiértékelés

A kiértékelő parancsok ugyanazt az elemzési folyamatot futtatják, mint az alkalmazás. Az `.onnx` modell az alkalmazás detektorával fut, a `.pt` modell az ultralytics-szel. A klipet először a test, majd a train könyvtárban keresik, és jelzik, ha a klip tanítóvideóról származik, mert az ilyen mérés nem független.

Az alapértelmezett értékeket a `research/run_config.json` adja, az alkalmazás ezt a fájlt nem olvassa. Az `active_clip_id` az alapértelmezett klip azonosítója, a `model_path` az alapértelmezett modell útvonala, a `confidence_threshold` a konfidenciaküszöb, a `show_tracking` pedig azt adja meg, hogy megjelenjenek-e a követést mutató ablakok.

**Modell futtatása egy klipen.**

```bash
uv run eval-inference [kapcsolók]
```

Futás közben ablakban mutatja a videót és a felülnézeti sávot a detekcióval, ez az Esc billentyűvel leállítható. A végén kiírja a detekciós arányt, az átlag- és a maximális sebességet, és megmutatja a sebességgörbét.

| Kapcsoló | Jelentés |
|---|---|
| `--clip <clip_id>` | A klip azonosítója, alapértelmezésben az `active_clip_id`. |
| `--lane <lane_id>` | Ez a sáv a klip saját sávja helyett. |
| `--model <path>` | A modell `.onnx` vagy `.pt` fájlja, alapértelmezésben a `model_path`. |
| `--confidence <threshold>` | A konfidenciaküszöb, alapértelmezésben a `confidence_threshold`. |
| `--no-show` | Nem nyit ablakot és nem rajzol grafikont, csak kiírja az eredményt. |

**A `.pt` és az `.onnx` modell összevetése.**

```bash
uv run eval-onnx [kapcsolók]
```

Ugyanazokon a képkockákon futtatja a `.pt` modellt és az ONNX-exportját, és kiírja, hány képkockán egyezik a detekció, mennyire fedik egymást a dobozok, mennyire tér el a konfidencia, az átlag- és a maximális sebesség, és mennyi a képkockánkénti futásidő. Ha az eltérés a megengedettnél nagyobb, 1-es hibakóddal lép ki.

| Kapcsoló | Jelentés |
|---|---|
| `--model <path>` | A modell `.pt` vagy `.onnx` fájlja, a párja a mellette lévő azonos nevű fájl. Alapértelmezésben a `model_path`. |
| `--clip <clip_id>` | A klip azonosítója, több klipre többször is megadható. Alapértelmezésben az `active_clip_id`. |
| `--confidence <threshold>` | A konfidenciaküszöb, alapértelmezésben a `confidence_threshold`. |

**A sebesség jelfeldolgozásának lépései.**

```bash
uv run eval-pipeline [kapcsolók]
```

Egy ábrán hasonlítja össze a sebességszámítás lépéseit egy klipen, a nyers deriválttól a mozgóátlagon és a Kalman-szűrőn át a végleges simított profilig. Az ábrát megjeleníti, és a `research/evaluation/output/pipeline_analysis_<clip_id>.png` fájlba menti.

| Kapcsoló | Jelentés |
|---|---|
| `--clip <clip_id>` | A klip azonosítója, alapértelmezésben az `active_clip_id`. |
| `--model <path>` | A modell `.onnx` vagy `.pt` fájlja, alapértelmezésben a `model_path`. |
| `--confidence <threshold>` | A konfidenciaküszöb, alapértelmezésben a `confidence_threshold`. |

**Kézi CSRT-követés.**

```bash
uv run csrt-track [kapcsolók]
```

A korábbi, modell nélküli kísérlet, a dolgozat referenciája. Az első képkockán egérrel kell bekeretezni az úszót, a kijelölést az Enter vagy a Szóköz véglegesíti, utána az OpenCV CSRT követője követi. A követés az Esc billentyűvel leállítható, a végén megjelenik a sebességgörbe.

| Kapcsoló | Jelentés |
|---|---|
| `--clip <clip_id>` | A klip azonosítója, alapértelmezésben az `active_clip_id`. |
| `--lane <lane_id>` | Ez a sáv a klip saját sávja helyett. |

#### Telepítő buildelése

A telepítő buildeléséhez az Inno Setup 6 kell, ez a `winget install JRSoftware.InnoSetup` paranccsal telepíthető. Ha nem az alapértelmezett helyre került, az `ISCC` környezeti változóban adható meg az `ISCC.exe` útvonala.

```bash
uv run --group build python packaging/build.py [kapcsolók]
```

A `--group build` a PyInstallert is telepíti a környezetbe. A build lépései a következők.

1. Összegyűjti a `packaging/release_models.json` modelljeinek `.onnx` és `.json` fájlját, a listában megadott megjelenített névvel és leírással.
2. A PyInstaller elkészíti a `packaging/dist/SwimmerTracker/` mappát a `SwimmerTracker.exe` programmal.
3. A smoke test a kész programot képernyő nélkül elindítja egy üres könyvtárral, és ellenőrzi, hogy a felület és minden modell betöltődik.
4. Az Inno Setup elkészíti a `packaging/dist/SwimmerTracker-<verzió>-setup.exe` telepítőt.

| Kapcsoló | Jelentés |
|---|---|
| `--skip-installer` | A PyInstaller build és a smoke test után leáll, telepítőt nem készít. |
| `--skip-smoke-test` | Kihagyja a smoke testet. |

A `packaging/release_models.json` minden bejegyzése egy `models/` mappabeli modell neve, a felhasználónak megjelenő név és egy rövid leírás. A verziószám a `pyproject.toml` `version` mezőjéből jön. A kész telepítőt érdemes kiadás előtt Windows Sandboxban kipróbálni.
