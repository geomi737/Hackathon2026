import os
import zipfile
import tempfile
import cv2
import numpy as np
import rasterio
import shutil


class ImageProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def normalize_band(self, band):
        """Нормализация канала в 0-255 uint8"""
        if band.dtype == 'uint16':
            band = (band / 256).astype('uint8')
        elif band.dtype == 'float32':
            if band.max() <= 1.0:
                band = (band * 255).astype('uint8')
            else:
                band = band.astype('uint8')
        return band

    def extract_layers(self, zip_path):
        """
        Извлекает слои SWIR, NDVI и True Color из ZIP архива.
        Возвращает пути к извлеченным/обработанным файлам.
        """
        layers = {
            'display_paths': {
                'true_color': None,
                'swir': None,
                'ndvi': None
            },
            'geo_paths': {
                'true_color': None,
                'swir': None,
                'ndvi': None
            },
            'original_name': os.path.basename(zip_path)
        }

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                namelist = z.namelist()

                # Эвристический поиск файлов
                true_file = next((n for n in namelist if any(
                    k in n.lower() for k in ['true', 'rgb', 'visual'])), None)
                swir_file = next((n for n in namelist if any(
                    k in n.lower() for k in ['swir', 'false', 'thermal', 'fire'])), None)
                ndvi_file = next(
                    (n for n in namelist if 'ndvi' in n.lower()), None)

                # Вспомогательная функция для извлечения и обработки
                def process_file(filename, key):
                    if not filename:
                        return None

                    extract_path = os.path.join(self.temp_dir, filename)
                    # Извлечение отдельного файла
                    z.extract(filename, self.temp_dir)

                    # Чтение и конвертация в PNG/Array для удобства использования
                    with rasterio.open(extract_path) as src:
                        # Логика конвертации в RGB/Grayscale PNG
                        if src.count >= 3:
                            img = src.read([1, 2, 3])
                            img = np.transpose(img, (1, 2, 0))  # HWC
                            # Нормализация
                            img = cv2.merge([self.normalize_band(img[:, :, 0]),
                                             self.normalize_band(img[:, :, 1]),
                                             self.normalize_band(img[:, :, 2])])
                        else:
                            img = src.read(1)
                            img = self.normalize_band(img)
                            if key == 'true_color':  # Принудительный RGB для YOLO
                                img = cv2.merge([img, img, img])

                        out_path = os.path.join(
                            self.temp_dir, f"{key}_{os.path.basename(filename)}.png")
                        # CV2 ожидает BGR
                        if len(img.shape) == 3:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        else:
                            img_bgr = img  # Grayscale

                        cv2.imwrite(out_path, img_bgr)
                        return out_path

                # extract_path ЭТО гео-путь (извлеченный TIF/JP2)
                layers['display_paths']['true_color'] = process_file(
                    true_file, 'true_color')
                layers['geo_paths']['true_color'] = os.path.join(
                    self.temp_dir, true_file) if true_file else None

                layers['display_paths']['swir'] = process_file(
                    swir_file, 'swir')
                layers['geo_paths']['swir'] = os.path.join(
                    self.temp_dir, swir_file) if swir_file else None

                layers['display_paths']['ndvi'] = process_file(
                    ndvi_file, 'ndvi')
                layers['geo_paths']['ndvi'] = os.path.join(
                    self.temp_dir, ndvi_file) if ndvi_file else None

        except Exception as e:
            print(f"Ошибка извлечения слоев: {e}")

        return layers

    def verify_fire(self, bboxes, swir_path, threshold):
        """
        Проверка обнаружений YOLO (пожары) с использованием слоя SWIR.
        bboxes: список [x1, y1, x2, y2]
        """
        verified_bboxes = []
        if not swir_path or not os.path.exists(swir_path):
            return []

        # Чтение SWIR как grayscale
        swir_img = cv2.imread(
            swir_path, cv2.IMREAD_GRAYSCALE)  # 0-255 нормализованный

        for box in bboxes:
            x1, y1, x2, y2 = box
            # Обрезка
            crop = swir_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            max_val = np.max(crop)
            if max_val > threshold:
                verified_bboxes.append({
                    'bbox': box,
                    'max_temp': max_val
                })

        return verified_bboxes

    def analyze_vegetation(self, ndvi_path):
        """
        Вычисляет средний NDVI.
        Предполагается вход 0-255, где 255 - высокая растительность.
        Возвращает: normalized_avg (0-1 approx).
        """
        if not ndvi_path or not os.path.exists(ndvi_path):
            return None

        img = cv2.imread(ndvi_path, cv2.IMREAD_GRAYSCALE)
        avg = np.mean(img)
        # Масштабирование к 0-1
        return avg / 255.0

    def resize_to_match(self, src_img, ref_img_shape):
        """
        Изменяет размер src_img, чтобы он соответствовал ref_img_shape (H, W).
        """
        h, w = ref_img_shape[:2]
        return cv2.resize(src_img, (w, h), interpolation=cv2.INTER_LINEAR)

    def match_histogram(self, source, reference):
        """
        Корректирует исходное изображение, чтобы оно соответствовало среднему и 
        стандартному отклонению эталонного изображения.
        """
        # Вычисление среднего и std
        mean_src = np.mean(source)
        std_src = np.std(source)
        mean_ref = np.mean(reference)
        std_ref = np.std(reference)

        # Нормализация источника
        if std_src == 0:
            return source  # Избегание деления на ноль

        # Result = (Source - Mean_Src) / Std_Src * Std_Ref + Mean_Ref
        corrected = (source - mean_src) / std_src * std_ref + mean_ref

        # Обрезка до допустимого диапазона [0, 1] для NDVI
        corrected = np.clip(corrected, 0, 1)
        return corrected

    def calculate_vegetation_loss(self, current_ndvi, historical_ndvi, threshold=0.2):
        """
        Вычисление потери растительности между историческим и текущим NDVI.
        Возвращает:
            mask: Бинарная маска зон потерь.
            bboxes: Список [x, y, w, h] зон потерь.
            contours: Список контуров для отрисовки.
        """
        if current_ndvi is None or historical_ndvi is None:
            return None, [], []

        # Обеспечение совпадения размеров
        if current_ndvi.shape != historical_ndvi.shape:
            historical_ndvi = self.resize_to_match(
                historical_ndvi, current_ndvi.shape)

        # 1. Радиометрическая коррекция (Histogram Matching)
        # Помогает избежать ложных срабатываний из-за разного освещения/атмосферы
        historical_calibrated = self.match_histogram(
            historical_ndvi, current_ndvi)

        # Разница
        # historical - current > threshold
        diff = historical_calibrated - current_ndvi

        # Создание маски
        loss_mask = (diff > threshold).astype(np.uint8) * 255

        # 2. Морфологическое подавление шума
        # Удаление мелких пятен (Шум)
        kernel_noise = np.ones((3, 3), np.uint8)
        loss_mask = cv2.morphologyEx(loss_mask, cv2.MORPH_OPEN, kernel_noise)

        # Объединение близлежащих компонентов
        kernel_merge = np.ones((15, 15), np.uint8)
        loss_mask = cv2.morphologyEx(loss_mask, cv2.MORPH_CLOSE, kernel_merge)

        # Поиск контуров
        contours, _ = cv2.findContours(
            loss_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # 3. Фильтр по площади контура
            # Игнорирование крошечных областей
            area = cv2.contourArea(cnt)
            if area > 50:  # Мин. площадь
                bboxes.append([x, y, w, h])

        return loss_mask, bboxes, contours
