import os
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch


class UnifiedDetector:
    def __init__(self, model_path='runs/detect/merged_det/weights/best.pt', conf_threshold=0.25):
        """
        Инициализация UnifiedDetector с моделью YOLO.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # Загрузка модели для уверенности, что она скачана и существует
        self.model = YOLO(self.model_path)

    def predict(self, image_path, use_sahi=True, conf_threshold=None, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        """
        Запуск инференса на изображении.

        Args:
            image_path (str): Путь к файлу изображения.
            use_sahi (bool): Использовать ли SAHI для нарезанного инференса.
            conf_threshold (float): Порог уверенности (переопределяет значение по умолчанию).

        Returns:
            list: Список обнаружений (bbox, score, class).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Определение используемого порога уверенности
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        if not use_sahi:
            # Стандартный инференс YOLO
            results = self.model(
                image_path, conf=conf, device=device)
            return results[0]  # Возвращаем первый результат
        else:
            # Инференс SAHI
            # Мы пересоздаем модель, чтобы гарантировать применение корректного порога,
            # так как AutoDetectionModel принимает его при инициализации.
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolo11',
                model_path=self.model_path,
                confidence_threshold=conf,
                device=device
            )

            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio
            )
            return result
