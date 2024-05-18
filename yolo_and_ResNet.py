import os
import shutil
import torch
from PIL import Image, ImageFile
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import timm
import cv2
from torchvision import transforms as T

# Разрешить загрузку поврежденных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Загрузка предобученной модели из timm
model_path = r"best_model_ResNet_20.pth"
model = timm.create_model('resnext50_32x4d.a3_in1k', pretrained=False)
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)
model.eval()

# Получение трансформаций модели (нормализация, изменение размера)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Классы животных
class_names = ["Кабарга", "Косуля", "Олень"]
class_indices = {"Кабарга": 0, "Косуля": 1, "Олень": 2}

def predict_image_class(image_path, model, processor, yolo_model, return_not_found=False):

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

            inputs = processor(cropped_img).unsqueeze(0)
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
            inputs = processor(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(inputs)

            logits = outputs
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

            predicted_class_idx = predicted_class_idx.item()

    return predicted_class_idx

# Function to distribute images into folders based on predictions
def distribute_images(input_folder, output_folder, model, processor, class_names):
    # Create output folders if they don't exist
    for class_name in class_names:
        class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    true_labels = []
    pred_labels = []
    file_names = []
    times = []

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


    # Process images from the input folder
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(input_folder, class_name)
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_folder, filename)
                try:
                    start_time = time.time()
                    predicted_class_idx = predict_image_class(image_path, model, processor, yolo_model)
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

# Function to calculate F1 score
def calculate_f1_score(true_labels, pred_labels):
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=1)
    return precision, recall, f1


if __name__ == "__main__":
    input_folder = r"C:/Users/user/Desktop/data/valid"
    output_folder = r"C:/Users/user/Desktop/sorted_predictions"
    # Распределение изображений по папкам и получение меток
    true_labels, pred_labels, file_names, average_time_per_image = distribute_images(input_folder, output_folder, model, transforms, class_names)

    # Вычисление F1-меры
    precision, recall, f1 = calculate_f1_score(true_labels, pred_labels)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average time per image: {average_time_per_image:.4f} seconds")

    