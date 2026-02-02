import re
import streamlit as st
import leafmap.foliumap as leafmap
import os
import cv2
import pandas as pd
import numpy as np
import plotly.express as px
from src.processor import ImageProcessor
from src.ml import UnifiedDetector
from src.geo import get_lat_lon, get_address_from_coords, get_bounds, pixels_to_coords
from src.sentinel import SentinelClient
import datetime
import matplotlib.pyplot as plt

# Конфигурация страницы
st.set_page_config(layout="wide", page_title="Technostart 2026 - SAD")
st.title("SAD: Satellite Anomaly Detection")
st.markdown(
    "Автоматическое обнаружение **Лесных пожаров**, **Нелегальных свалок** и **Потери растительности**.")

# Настройки боковой панели
st.sidebar.header("Настройки")
conf_threshold = st.sidebar.slider(
    "Порог уверенности", 0.0, 1.0, 0.25, 0.05)
thermal_threshold = st.sidebar.number_input(
    "Тепловой порог (0-255)", value=100)
veg_loss_threshold = st.sidebar.slider(
    "Порог потери растительности", 0.0, 1.0, 0.2, 0.05)
debug_mode = st.sidebar.checkbox("Режим отладки", value=False)

# Загрузка модели


@st.cache_resource
def load_models():
    model_path = "runs/detect/merged_det/weights/best.pt"
    if not os.path.exists(model_path):
        st.warning(
            f"Модель не найдена по пути {model_path}, используется стандартная 'yolo11s.pt'")
        return UnifiedDetector(model_path="yolo11s.pt", conf_threshold=conf_threshold)
    return UnifiedDetector(model_path=model_path, conf_threshold=conf_threshold)


detector = load_models()
processor = ImageProcessor()

# Загрузчик файлов (ТОЛЬКО ZIP)
uploaded_files = st.sidebar.file_uploader(
    "Загрузите 3-слойный ZIP-архив (True Color, NDVI, SWIR)",
    type=['zip'],
    accept_multiple_files=True
)

all_detections = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"📦 Анализ: {uploaded_file.name}", expanded=True):
            # Временное сохранение ZIP
            temp_zip_path = os.path.join(
                processor.temp_dir, uploaded_file.name)
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 1. Извлечение слоев
            layers = processor.extract_layers(temp_zip_path)

            display_paths = layers['display_paths']
            geo_paths = layers['geo_paths']

            true_path = display_paths.get('true_color')
            swir_path = display_paths.get('swir')
            ndvi_path = display_paths.get('ndvi')

            geo_true_path = geo_paths.get('true_color')

            if not true_path:
                st.error(
                    "Отсутствует слой 'True Color' в ZIP. Невозможно продолжить.")
                continue

            # Получение размеров изображения для дальнейшего использования
            img_h, img_w = cv2.imread(true_path).shape[:2]

            # 2. Обнаружение пожаров и свалок (YOLO на True Color)
            # КЭШ РЕЗУЛЬТАТА чтобы избежать повторного запуска при переключении слоев
            cache_key = f"pred_{uploaded_file.name}_{conf_threshold}"

            if cache_key in st.session_state:
                results = st.session_state[cache_key]
            else:
                with st.spinner("Поиск аномалий..."):
                    results = detector.predict(
                        true_path,
                        conf_threshold=conf_threshold,
                        slice_height=640, slice_width=640
                    )
                    st.session_state[cache_key] = results

            # 3. Обработка обнаружений
            # Класс 0: Пожар, Класс 1: Свалка
            verified_detections = []  # Список {bbox, type, lat, lon, address}

            # --- Извлечение "сырых" рамок ---
            raw_fire = []
            raw_waste = []

            if hasattr(results, 'object_prediction_list'):  # SAHI
                for p in results.object_prediction_list:
                    if p.score.value >= conf_threshold:
                        cat_name = p.category.name.lower()
                        if 'fire' in cat_name or p.category.id == 0:
                            bbox = p.bbox
                            raw_fire.append([int(bbox.minx), int(
                                bbox.miny), int(bbox.maxx), int(bbox.maxy)])
                        elif 'waste' in cat_name or p.category.id == 1:
                            bbox = p.bbox
                            raw_waste.append([int(bbox.minx), int(
                                bbox.miny), int(bbox.maxx), int(bbox.maxy)])
            else:  # Стандартный YOLO
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = box.conf[0]
                    if conf >= conf_threshold:
                        if cls_id == 0:  # Fire
                            raw_fire.append(list(map(int, box.xyxy[0])))
                        elif cls_id == 1:  # Waste
                            raw_waste.append(list(map(int, box.xyxy[0])))

            # --- Проверка пожаров ---
            # Verified fire returns list of {bbox, max_temp}
            verified_fire_data = processor.verify_fire(
                raw_fire, swir_path, thermal_threshold)

            # --- Компиляция финального списка ---
            # Пожары
            for item in verified_fire_data:
                cx = (item['bbox'][0] + item['bbox'][2]) / 2
                cy = (item['bbox'][1] + item['bbox'][3]) / 2
                lat, lon = get_lat_lon(geo_true_path, cx, cy)
                addr = get_address_from_coords(
                    lat, lon) if lat else "Неизвестно"

                verified_detections.append({
                    "bbox": item['bbox'],
                    "type": "wildfire",
                    "lat": lat, "lon": lon,
                    "address": addr,
                    "details": f"Temp: {item['max_temp']}"
                })

            # Свалки (без верификации)
            for box in raw_waste:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                lat, lon = get_lat_lon(geo_true_path, cx, cy)
                addr = get_address_from_coords(
                    lat, lon) if lat else "Неизвестно"

                verified_detections.append({
                    "bbox": box,
                    "type": "waste",
                    "lat": lat, "lon": lon,
                    "address": addr,
                    "details": "Confidence High"
                })

            # 4. Анализ растительности и анализ исторических различий (HDA)
            current_ndvi = None
            if ndvi_path:
                current_ndvi_img = cv2.imread(ndvi_path, cv2.IMREAD_GRAYSCALE)
                current_ndvi = processor.analyze_vegetation(ndvi_path)

            tree_status = "Неизвестно"
            loss_detected = False

            sentinel_client = SentinelClient()
            historical_data = None
            diff_mask = None
            vegetation_loss_boxes = []  # [x,y,w,h]
            vegetation_loss_contours = []

            # Логика HDA
            # Отладочные учетные данные
            if debug_mode:
                st.write(
                    f"DEBUG: Client ID present: {bool(sentinel_client.config.sh_client_id)}")
                st.write(
                    f"DEBUG: Client Secret present: {bool(sentinel_client.config.sh_client_secret)}")
                st.write(f"DEBUG: Config: {sentinel_client.config}")

            if sentinel_client.config.sh_client_id and sentinel_client.config.sh_client_secret:
                hda_enabled = True

                # Получение исторических данных
                # Попытка найти дату в имени файла (ZIP или внутренний файл)
                photo_date = None

                # Кандидаты для поиска даты
                date_candidates = [uploaded_file.name]
                if layers['geo_paths'].get('true_color'):
                    date_candidates.append(os.path.basename(
                        layers['geo_paths']['true_color']))

                for filename in date_candidates:
                    if not filename:
                        continue

                    # Паттерн 1: YYYY-MM-DD
                    match_iso = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
                    # Паттерн 2: YYYYMMDD
                    match_compact = re.search(
                        r'(\d{4})(\d{2})(\d{2})', filename)

                    if match_iso:
                        photo_date = datetime.date(int(match_iso.group(1)), int(
                            match_iso.group(2)), int(match_iso.group(3)))
                        break
                    elif match_compact:
                        photo_date = datetime.date(int(match_compact.group(1)), int(
                            match_compact.group(2)), int(match_compact.group(3)))
                        break

                if photo_date:
                    st.info(f"Обнаруженная дата снимка: {photo_date}")
                    target_date = photo_date - datetime.timedelta(days=365)
                else:
                    st.warning(
                        "Не удалось найти дату в имени файла (YYYY-MM-DD или YYYYMMDD). Используется СЕГОДНЯ как эталон.")
                    target_date = datetime.date.today() - datetime.timedelta(days=365)

                # Получение границ
                bounds = get_bounds(geo_true_path)  # (minx, miny, maxx, maxy)

                if bounds:
                    cache_key_hda = f"hda_{uploaded_file.name}_{target_date}"
                    if cache_key_hda in st.session_state:
                        historical_data = st.session_state[cache_key_hda]
                    else:
                        st.caption(
                            f"Получение данных Sentinel за {target_date}...")

                        historical_data = sentinel_client.fetch_historical_data(
                            bounds, img_w, img_h, target_date
                        )
                        st.write("DEBUG: Historical data fetched or None.")
                        if historical_data is None:
                            st.warning(
                                "Sentinel API не вернул данных. Проверьте ключи или охват территории.")
                        else:
                            st.write(
                                "DEBUG: Data valid. Updating session state.")

                        st.session_state[cache_key_hda] = historical_data

                    if historical_data:
                        # Анализ изменений
                        # Нам нужны текущий NDVI (0-1 float) и исторический NDVI (0-1 float)
                        # current_ndvi_img это 0-255. Конвертируем в 0-1.
                        curr_ndvi_norm = current_ndvi_img.astype(
                            np.float32) / 255.0

                        # НОРМАЛИЗАЦИЯ ИСТОРИЧЕСКИХ ДАННЫХ для соответствия диапазону 0-1 (убираем отрицательные значения)
                        hist_ndvi_raw = historical_data['ndvi']
                        hist_ndvi_norm = np.clip(hist_ndvi_raw, 0, 1)

                        # DEBUG STATS for Calibration
                        st.subheader("Статистика калибровки данных")
                        c_stat1, c_stat2 = st.columns(2)
                        with c_stat1:
                            st.write(f"**Текущий NDVI (Загруженный)**")
                            st.write(f"Min: {curr_ndvi_norm.min():.2f}")
                            st.write(f"Max: {curr_ndvi_norm.max():.2f}")
                            st.write(f"Mean: {curr_ndvi_norm.mean():.2f}")

                        with c_stat2:
                            st.write(f"**Исторический NDVI (Sentinel)**")
                            st.write(f"Min: {hist_ndvi_norm.min():.2f}")
                            st.write(f"Max: {hist_ndvi_norm.max():.2f}")
                            st.write(f"Mean: {hist_ndvi_norm.mean():.2f}")

                        diff_mask, vegetation_loss_boxes, vegetation_loss_contours = processor.calculate_vegetation_loss(
                            curr_ndvi_norm, hist_ndvi_norm, threshold=veg_loss_threshold
                        )

                        if len(vegetation_loss_boxes) > 0:
                            loss_detected = True
                            tree_status = f"ОБНАРУЖЕНА КРИТИЧЕСКАЯ ПОТЕРЯ ({len(vegetation_loss_boxes)} зон)"
                        else:
                            tree_status = "Стабильно (по сравнению с прошлым годом)"

            # Сбор обнаружений потери растительности
            # Конвертация рамок в lat/lon и добавление в глобальный список
            if loss_detected:
                for box in vegetation_loss_boxes:
                    x, y, w, h = box
                    cx, cy = x + w/2, y + h/2
                    lat, lon = get_lat_lon(geo_true_path, cx, cy)
                    # SKIP ADDRESS LOOKUP FOR VEGETATION LOSS TO PREVENT HANGING
                    # addr = get_address_from_coords(lat, lon) if lat else "Unknown"
                    addr = "Зона потери растительности"

                    all_detections.append({
                        "file": uploaded_file.name,
                        "latitude": lat,
                        "longitude": lon,
                        "problem": "vegetation_loss",
                        "address": addr
                    })

            # Добавление остальных обнаружений в глобальный список
            for det in verified_detections:
                all_detections.append({
                    "file": uploaded_file.name,
                    "latitude": det['lat'],
                    "longitude": det['lon'],
                    "problem": det['type'],
                    "address": det['address']
                })

            # 5. UI Макет
            c1, c2 = st.columns([2, 1])

            with c1:
                # Выбор слоя
                layer_options = [
                    "True Color", "Проверка пожаров (SWIR)", "Растительность (NDVI)"]
                if historical_data:
                    layer_options.extend(
                        ["Historical True Color", "Historical SWIR", "Historical NDVI"])

                layer_choice = st.radio(
                    "Выберите слой:", layer_options, horizontal=True, label_visibility="collapsed", key=f"layer_{uploaded_file.name}")

                if layer_choice == "True Color":
                    img_viz = cv2.imread(true_path)
                    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)

                    # === ОТРИСОВКА ПОТЕРИ РАСТИТЕЛЬНОСТИ ===
                    if loss_detected:
                        overlay = img_viz.copy()
                        # Желтый цвет заливки (RGB: 255, 255, 0)
                        cv2.drawContours(
                            overlay, vegetation_loss_contours, -1, (255, 255, 0), -1)
                        # Смешивание с оригиналом (alpha blend)
                        cv2.addWeighted(overlay, 0.4, img_viz, 0.6, 0, img_viz)
                        # Обводка контуров для четкости (Желтый)
                        cv2.drawContours(
                            img_viz, vegetation_loss_contours, -1, (255, 255, 0), 2)

                elif layer_choice == "Проверка пожаров (SWIR)":
                    img_viz = cv2.imread(swir_path) if swir_path else np.zeros(
                        (100, 100, 3), np.uint8)
                    if len(img_viz.shape) == 2:
                        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
                    else:
                        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)

                elif layer_choice == "Растительность (NDVI)":
                    img_viz = cv2.imread(ndvi_path) if ndvi_path else np.zeros(
                        (100, 100, 3), np.uint8)

                    # Если изображение в градациях (или эффективно ч/б), применяем Colormap
                    if len(img_viz.shape) == 3:
                        # Проверка, совпадают ли каналы
                        if np.allclose(img_viz[:, :, 0], img_viz[:, :, 1]) and np.allclose(img_viz[:, :, 1], img_viz[:, :, 2]):
                            img_gray = img_viz[:, :, 0]
                            img_viz = cv2.applyColorMap(
                                img_gray, cv2.COLORMAP_JET)
                            img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
                        else:
                            img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
                    else:
                        # Прямое чтение градаций
                        img_viz = cv2.applyColorMap(img_viz, cv2.COLORMAP_JET)
                        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)

                elif layer_choice == "Historical True Color" and historical_data:
                    img_viz = historical_data['true_color']
                elif layer_choice == "Historical SWIR" and historical_data:
                    img_viz = historical_data['swir']
                elif layer_choice == "Historical NDVI" and historical_data:
                    # NDVI это single channel float (-1 to 1).
                    # Используем Matplotlib 'YlGn' (Yellow-Green)
                    ndvi_data = historical_data['ndvi']
                    norm = plt.Normalize(vmin=0, vmax=0.8)
                    cmap = plt.get_cmap('YlGn')

                    # Применяем colormap (возвращает RGBA float 0-1)
                    ndvi_colored = cmap(norm(ndvi_data))

                    # Конвертация в uint8 RGB (без Alpha)
                    img_viz = (ndvi_colored[:, :, :3] * 255).astype(np.uint8)

                else:
                    img_viz = np.zeros((300, 300, 3), np.uint8)

                # Отрисовка рамок YOLO (Пожар/Свалка)
                if "Historical" not in layer_choice:
                    for det in verified_detections:
                        col = (255, 0, 0) if det['type'] == 'wildfire' else (
                            255, 255, 0)  # Red or Yellow
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(img_viz, (x1, y1), (x2, y2), col, 3)

                fig = px.imshow(img_viz)
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Интерактивная карта")
                m = leafmap.Map(draw_control=False,
                                layers_control=True, measure_control=False)
                # Маркеры
                for det in verified_detections:
                    if det['lat'] and det['lon']:
                        m.add_marker(location=[det['lat'], det['lon']],
                                     popup=f"{det['type'].upper()}: {det['address']}")

                # Маркеры потери леса
                if loss_detected:
                    for cnt in vegetation_loss_contours:
                        # Упрощение контура (epsilon = 1% от длины дуги)
                        epsilon = 0.01 * cv2.arcLength(cnt, True)
                        approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)

                        # Конвертация: [ [[x,y]], ... ] -> [ [x,y], ... ]
                        pts = approx_cnt.reshape(-1, 2).tolist()

                        # Преобразование в Lat/Lon
                        geo_coords = pixels_to_coords(geo_true_path, pts)

                        if geo_coords and len(geo_coords) > 2:
                            # Leafmap ожидает [[lat, lon], ...] для полигонов, если через folium/ipyleaflet
                            # Но pixels_to_coords возвращает [lon, lat] (GeoJSON стандарт)
                            # Проверим документацию add_polygon / add_geojson.
                            # Ensure closed loop
                            if geo_coords[0] != geo_coords[-1]:
                                geo_coords.append(geo_coords[0])

                            geojson_data = {
                                "type": "FeatureCollection",
                                "features": [
                                    {
                                        "type": "Feature",
                                        "geometry": {
                                            "type": "Polygon",
                                            "coordinates": [geo_coords]
                                        },
                                        "properties": {}
                                    }
                                ]
                            }

                            style = {
                                "stroke": True,
                                "color": "yellow",
                                "weight": 2,
                                "opacity": 1,
                                "fill": True,
                                "fillColor": "yellow",
                                "fillOpacity": 0.4,
                            }

                            m.add_geojson(
                                geojson_data, layer_name="Vegetation Loss", style_function=lambda x: style)

                # Центрирование
                center_lat, center_lon = get_lat_lon(
                    geo_true_path, img_w/2, img_h/2)
                if center_lat and center_lon:
                    m.set_center(center_lon, center_lat, zoom=13)

                m.to_streamlit(height=400)

                st.divider()
                st.write(f"**Состояние растительности:** {tree_status}")
                if loss_detected:
                    st.error("ОБНАРУЖЕНА ПОТЕРЯ")

# Экспорт CSV
if all_detections:
    st.divider()
    st.header("Отчет о нарушениях")
    df = pd.DataFrame(all_detections)
    # Порядок колонок
    cols = ["file", "latitude", "longitude", "problem", "address"]
    # Фильтрация валидных колонок
    final_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[final_cols])

    st.download_button(
        "Скачать полный отчет (CSV)",
        df[final_cols].to_csv(index=False),
        "violations_report.csv"
    )
