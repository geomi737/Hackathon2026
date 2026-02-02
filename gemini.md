# 🚀 PROJECT: Technostart 2026 Hackathon Entry
**Role:** Hackabud (AI Teammate) & User (Developer)
**Goal:** Разработка сервиса для автоматического мониторинга земной поверхности, обнаружения аномалий (свалки, вырубки, пожары) и их визуализации на интерактивной карте.
**Core Concept:** "Анализатор фото со спутников"

---

## 🛠 TECH STACK (Architecture)
Мы используем **Python-based** подход с акцентом на скорость разработки и SOTA решения.

| Компонент | Библиотека | Назначение |
| :--- | :--- | :--- |
| **ML Core** | `Ultralytics YOLOv11` | Детекция объектов (Object Detection). Быстрая и точная. |
| **Preprocessing** | `SAHI` | Slicing Aided Hyper Inference. Нарезка больших спутниковых снимков на чанки для детекции мелких объектов. |
| **Geo-Logic** | `Rasterio` | Работа с GeoTIFF. Конвертация пиксельных координат (bbox) в географические (Lat/Lon). |
| **Geocoding** | `Geopy` (Nominatim) | Обратное геокодирование. Преобразование координат в человеческий адрес (Адрес, Город). |
| **Frontend** | `Streamlit` | Быстрый веб-интерфейс (Pure Python). |
| **Visualization** | `Leafmap` | Интерактивная карта, работа со слоями, маркерами и гео-аналитикой. |
| **ML Interaction** | `pytorch` | Взаимодействие, инференс и обучение YOLOv11. |
---

## 🚦 ROADMAP & STAGES

### 🔴 STAGE 1: MVP (Critical / Day 1)
*Задача: Сделать рабочий прототип, который "глотает" GeoTIFF и показывает точку на карте.*
- [X] **Data:** Найти датасет (свалки/пожары/вырубки) и разметить/подготовить.
- [X] **ML Baseline:** Обучить YOLOv8/v11.
- [X] **Geo Script:** Написать пайплайн: `Image -> SAHI -> YOLO -> Rasterio -> Lat/Lon`.
- [X] **UI Base:** Поднять Streamlit + Leafmap. Отобразить найденную точку на базовой карте.
- [X] **Current status download** Скачивание текущего статуса (True Color, NVDI, SWIR) этой же области

### 🟡 STAGE 2: ENHANCEMENTS (Quality / Night-Day 2)
*Задача: Превратить прототип в продукт.*
- [X] **Address Lookup:** Интеграция `Geopy`. Вывод адреса.
- [X] **Multi-class:** Разделение на классы (Пожар=Красный, Потеря леса=Зеленый).
- [X] **Reporting:** Выгрузка списка нарушений в CSV/Excel/PDF.
- [X] **Zoom** Зум картинки
- [X] **MultiAnalyze** Возможность загружать несколько фоток за раз
- [X] **MultilayerEye** Возможность смотреть разные слои

### 🟢 STAGE 3: "WOW" FEATURES (Victory / Final Polish)
*Задача: Удивить жюри.*
- [ ] Успеть доделать
---

## 🔄 PIPELINE (Data Flow)
1. **User Input:** Upload GeoTIFF (`.tif`) via Streamlit.
2. **Tiling:** SAHI slices image into $640 \times 640$ patches.
3. **Inference:** YOLO detects anomalies on patches.
4. **Merge:** SAHI merges detections back to original image size.
5. **Georeferencing:** Rasterio maps pixel $(x,y)$ to $(lat, lon)$.
6. **Enrichment:** Geopy fetches address string for $(lat, lon)$.
7. **Output:** Leafmap renders markers with metadata on interactive map.

---

## 📝 CURRENT STATUS / CONTEXT LOG
*Update this section to keep track of progress.*
* **Current Focus:** Polish programm
* **Blockers:** None.