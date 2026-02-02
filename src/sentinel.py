import os
import datetime
import numpy as np
from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox
)
from dotenv import load_dotenv

load_dotenv()


class SentinelClient:
    def __init__(self):
        # Получение ключей из .env
        self.client_id = os.getenv("sentinelid")
        self.client_secret = os.getenv("sentinelkey")

        self.config = SHConfig()

        # Проверка наличия ключей
        if self.client_id and self.client_secret:
            self.config.sh_client_id = self.client_id
            self.config.sh_client_secret = self.client_secret
        else:
            print("⚠️ Ошибка: API ключи Sentinel не найдены!")

    def fetch_historical_data(self, bbox_coords, width, height, target_date):
        """
        Получение исторических данных Sentinel-2.
        bbox_coords: [LON_MIN, LAT_MIN, LON_MAX, LAT_MAX]
        """
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

        # Интервал времени: +/- 30 дней от целевой даты для поиска безоблачных снимков
        time_interval = (
            target_date - datetime.timedelta(days=30),
            target_date + datetime.timedelta(days=30)
        )

        # Evalscript для получения необходимых каналов
        raw_evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08", "B12"],
                output: { bands: 5, sampleType: "FLOAT32" }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02, sample.B08, sample.B12]; 
        }
        """

        try:
            request = SentinelHubRequest(
                evalscript=raw_evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        maxcc=0.2,  # Фильтр: Облачность не более 20%
                        mosaicking_order="leastCC"  # Выбор пикселей с наименьшей облачностью
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response(
                        "default", MimeType.TIFF)
                ],
                bbox=bbox,
                size=(width, height),
                config=self.config
            )

            # Получение данных
            response = request.get_data()

            if not response or len(response) == 0:
                print("❌ Sentinel API: Нет снимков без облаков в этот период.")
                return None

            # Выбор первого доступного снимка (Sentinel сортирует от старых к новым по умолчанию)
            # Так как мы используем mosaicking_order="leastCC", мы получим наилучший доступный снимок.
            data = response[0]

            print(f"✅ Получен снимок. Размер: {data.shape}")

            # Нормализация и обработка каналов
            # Sentinel L2A FLOAT32 значения обычно от 0.0 до 1.0 (умножаем для визуализации)

            # True Color (RGB)
            r = np.clip(data[:, :, 0] * 2.5 * 255, 0, 255).astype(np.uint8)
            g = np.clip(data[:, :, 1] * 2.5 * 255, 0, 255).astype(np.uint8)
            b = np.clip(data[:, :, 2] * 2.5 * 255, 0, 255).astype(np.uint8)
            true_color = np.dstack((r, g, b))

            # NDVI (Normalized Difference Vegetation Index)
            nir = data[:, :, 3]
            red = data[:, :, 0]

            # Защита от деления на ноль
            denominator = (nir + red)
            ndvi = np.divide(
                (nir - red),
                denominator,
                out=np.zeros_like(nir),
                where=denominator != 0
            )

            # SWIR (Short-Wave Infrared) Composite
            # R=SWIR (B12), G=NIR (B08), B=Red (B04)
            swir_band = data[:, :, 4]  # B12

            swir_vis_r = np.clip(swir_band * 2.5 * 255,
                                 0, 255).astype(np.uint8)
            swir_vis_g = np.clip(nir * 2.5 * 255, 0, 255).astype(np.uint8)
            swir_vis_b = np.clip(red * 2.5 * 255, 0, 255).astype(np.uint8)

            swir_composite = np.dstack((swir_vis_r, swir_vis_g, swir_vis_b))

            return {
                "true_color": true_color,
                "ndvi": ndvi,
                "swir": swir_composite
            }

        except Exception as e:
            print(f"🔥 Ошибка Sentinel API: {e}")
            return None
