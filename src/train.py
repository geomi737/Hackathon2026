from ultralytics import YOLO
import os


def train_model(epochs):
    print("Запуск обучения модели (Обнаружение Свалок и Пожаров)...")

    # Проверка существующего чекпоинта
    # Стандартный путь YOLO: project/name/weights/last.pt
    model_path = "runs/detect/merged_det/weights/last.pt"

    if os.path.exists(model_path):
        print(f"Возобновление обучения с {model_path}...")
        model = YOLO(model_path)
    else:
        print("Запуск нового обучения...")
        model = YOLO("yolo11s")
        # Загрузка модели детекции (Small, COCO-pretrained)
    # Обучение
    results = model.train(
        data="dataset/merged/data.yaml",
        epochs=epochs,
        imgsz=640,
        name="merged_det",
        exist_ok=True
    )
    print("Обучение завершено.")
    return model


if __name__ == "__main__":
    try:
        epochs = int(input("Введите количество эпох: "))
    except ValueError:
        print("Неверный ввод. По умолчанию 1 эпоха.")
        epochs = 1

    # Убедимся, что директория runs существует
    os.makedirs("runs/detect", exist_ok=True)

    try:
        train_model(epochs)
    except Exception as e:
        print(f"Ошибка обучения: {e}")
