import os
import shutil
import torch
from PIL import Image, ImageFile
import time
import cv2


# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def predict_image_class_resnet(image_path, model, processor, yolo_model, device, return_not_found=False):
    # Загрузка изображения
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Файл изображения не найден по пути {image_path}")

    # Конвертация изображения из BGR в RGB
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Конвертация изображения из numpy в PIL
    img = Image.fromarray(image_cv)

    # Запуск модели YOLOv5 на изображении
    results = yolo_model(img)

    # Извлечение bounding boxes
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    coordinates = results.xyxyn[0][:, :-1].cpu().numpy()

    # Классы животных в COCO: 16 - кошка, 17 - собака
    animal_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

    # Фильтрация только животных
    animal_boxes = [coordinates[i] for i in range(len(labels)) if int(labels[i]) in animal_classes]

    predictions = []
    weights = []

    # Если обнаружены животные
    if len(animal_boxes) > 0:
        for box in animal_boxes:
            x1, y1, x2, y2 = box[:4]
            x1, y1, x2, y2 = int(x1 * img.width), int(y1 * img.height), int(x2 * img.width), int(y2 * img.height)
            cropped_img = img.crop((x1, y1, x2, y2))

            inputs = processor(cropped_img).unsqueeze(0).to(device)  # Перемещаем тензор на нужное устройство
            with torch.no_grad():
                outputs = model(inputs)

            logits = outputs
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

            predictions.append(predicted_class_idx.item())
            weights.append(confidence.item())

        # Выполнение взвешенного голосования
        weighted_votes = {}
        for i, prediction in enumerate(predictions):
            if prediction not in weighted_votes:
                weighted_votes[prediction] = 0
            weighted_votes[prediction] += weights[i]

        predicted_class_idx = max(weighted_votes, key=weighted_votes.get)
    else:
        if return_not_found:
            return 3
        else:
            # Если животные не обнаружены, классифицируем целое изображение
            inputs = processor(img).unsqueeze(0).to(device)  # Перемещаем тензор на нужное устройство
            with torch.no_grad():
                outputs = model(inputs)

            logits = outputs
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

            predicted_class_idx = predicted_class_idx.item()

    return predicted_class_idx

# Function to distribute images into folders based on predictions
def distribute_images_val_resnet(input_folder, output_folder, model, processor, class_names, device):
    # Create output folders if they don't exist
    for class_name in class_names:
        class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    true_labels = []
    pred_labels = []
    file_names = []
    times = []

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

    # Process images from the input folder
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(input_folder, class_name)
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_folder, filename)
                try:
                    start_time = time.time()
                    predicted_class_idx = predict_image_class_resnet(image_path, model, processor, yolo_model, device)
                    end_time = time.time()

                    predicted_class_name = class_names[predicted_class_idx]
                    shutil.copy(image_path, os.path.join(output_folder, predicted_class_name, filename))
                    print(f"Copied {filename} to {predicted_class_name} folder")
                    
                    true_labels.append(class_idx)
                    pred_labels.append(predicted_class_idx)
                    file_names.append(filename)
                    times.append(end_time - start_time)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    average_time_per_image = sum(times) / len(times)
    return true_labels, pred_labels, file_names, average_time_per_image


def distribute_images_pred_resnet(input_folder, output_folder, model, processor, class_names, device):
    # Создание выходных папок, если они не существуют
    for class_name in class_names:
        class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    # Загружаем YOLOv5 модель и перемещаем на нужное устройство
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

    # Обработка изображений из входной папки
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            try:
                start_time = time.time()
                predicted_class_idx = predict_image_class_resnet(image_path, model, processor, yolo_model, device, return_not_found=False)
                end_time = time.time()

                predicted_class_name = class_names[predicted_class_idx]
                shutil.copy(image_path, os.path.join(output_folder, predicted_class_name, filename))
                print(f"Скопировано {filename} в папку {predicted_class_name}")
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

    print("Распределение изображений завершено.")


def distribute_images_val_efficientnet(input_folder, output_folder, model, preprocess, class_names, device):
    # Create output folders if they don't exist
    for class_name in class_names:
        class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    true_labels = []
    pred_labels = []
    file_names = []
    times = []

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

    # Process images from the input folder
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(input_folder, class_name)
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_folder, filename)
                try:
                    start_time = time.time()
                    predicted_class_idx = predict_image_class_efficientnet(image_path, model, preprocess, yolo_model, device)
                    end_time = time.time()

                    predicted_class_name = class_names[predicted_class_idx]
                    shutil.copy(image_path, os.path.join(output_folder, predicted_class_name, filename))
                    print(f"Copied {filename} to {predicted_class_name} folder")

                    true_labels.append(class_idx)
                    pred_labels.append(predicted_class_idx)
                    file_names.append(filename)
                    times.append(end_time - start_time)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    average_time_per_image = sum(times) / len(times)
    return true_labels, pred_labels, file_names, average_time_per_image


def predict_image_class_efficientnet(image_path, model, preprocess, yolo_model, device, return_not_found=False):
    # Загрузка изображения
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Файл изображения не найден по пути {image_path}")

    # Конвертация изображения из BGR в RGB
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Конвертация изображения из numpy в PIL
    img = Image.fromarray(image_cv)

    # Запуск модели YOLOv5 на изображении
    results = yolo_model(img)

    # Извлечение bounding boxes
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    coordinates = results.xyxyn[0][:, :-1].cpu().numpy()

    # Классы животных в COCO: 16 - кошка, 17 - собака
    animal_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

    # Фильтрация только животных
    animal_boxes = [coordinates[i] for i in range(len(labels)) if int(labels[i]) in animal_classes]

    predictions = []
    weights = []

    # Если обнаружены животные
    if len(animal_boxes) > 0:
        for box in animal_boxes:
            x1, y1, x2, y2 = box[:4]
            x1, y1, x2, y2 = int(x1 * img.width), int(y1 * img.height), int(x2 * img.width), int(y2 * img.height)
            cropped_img = img.crop((x1, y1, x2, y2))

            image = cropped_img
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)

            logits = outputs
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

            predictions.append(predicted_class_idx.item())
            weights.append(confidence.item())

        # Выполнение взвешенного голосования
        weighted_votes = {}
        for i, prediction in enumerate(predictions):
            if prediction not in weighted_votes:
                weighted_votes[prediction] = 0
            weighted_votes[prediction] += weights[i]

        predicted_class_idx = max(weighted_votes, key=weighted_votes.get)
    else:
        if return_not_found:
            return 3
        else:
            # Если животные не обнаружены, классифицируем целое изображение
            image = img
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image)

            logits = outputs
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

            predicted_class_idx = predicted_class_idx.item()

    return predicted_class_idx


def distribute_images_pred_efficientnet(input_folder, output_folder, model, preprocess, class_names, device):
    # Создание выходных папок, если они не существуют
    for class_name in class_names:
        class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    # Загружаем YOLOv5 модель и перемещаем на нужное устройство
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

    # Обработка изображений из входной папки
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            try:
                start_time = time.time()
                predicted_class_idx = predict_image_class_efficientnet(image_path, model, preprocess, yolo_model, device, return_not_found=False)
                end_time = time.time()

                predicted_class_name = class_names[predicted_class_idx]
                shutil.copy(image_path, os.path.join(output_folder, predicted_class_name, filename))
                print(f"Скопировано {filename} в папку {predicted_class_name}")
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

    print("Распределение изображений завершено.")
