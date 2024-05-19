
## Настройка контейнера Docker и операции с моделями

### 1. Сборка контейнера Docker

```sh
docker build -t your_image_name .
```

### 2. Запуск контейнера Docker

При необходимости смонтируйте папку с данными:

```sh
docker run --gpus all -it -v /путь/к/вашей/папке:/workspace/mounted_folder your_image_name
```

**Пример:**

```sh
docker run -it -v C:/valid:/workspace/mounted_folder your_image_name
```

### 3. Запуск валидации

```sh
python ResNet_eval.py C:/Users/user/Desktop/data/valid C:/Users/user/Desktop/sorted_predictions --model_path best_model_ResNet_20.pth --device cuda
```

**Пример:**

```sh
python ResNet_eval.py mounted_folder/valid mounted_folder/sorted_predictions --model_path mounted_folder/best_model_ResNet_20.pth --device cuda
```

### 4. Запуск обучения

```sh
python ResNet_train.py C:/Users/user/Desktop/data/train C:/Users/user/ack/artiodactyls_classification/best_model_ResNet_20.pth 10 --device cpu
```

python ResNet_train_2.py C:/Users/user/Desktop/data/train/Кабала C:/Users/user/Desktop/data/train/Косуля C:/Users/user/Desktop/data/train/Олень C:/Users/user/Downloads/best_model_ResNet_20.pth 10 --device cuda

**Пример:**

```sh
python ResNet_train.py mounted_folder/valid mounted_folder/best_model_ResNet_20.pth 10 --device cpu
```

### 5. Запуск предсказания

```sh
python ResNet_pred.py C:/Users/user/Desktop/test_valid C:/Users/user/Desktop/sorted_predictions --model_path best_model_ResNet_20.pth --device cpu
```

**Пример:**

```sh
python ResNet_pred.py mounted_folder/valid/Кабарга mounted_folder/sorted_predictions --model_path mounted_folder/best_model_ResNet_20.pth --device cpu
```

### Ссылки на обученные модели

- [Модель ResNet](https://drive.google.com/file/d/1TTkXIEqNbkzgpsFuLDUzvikLnzqJFay_/view?usp=sharing)
- [Модель EfficientNet](https://drive.google.com/file/d/1xW7QU0E-qn7ZiDFAFLB9lEJP_9nl0gcR/view?usp=sharing)
- [Модель YOLOv5s](https://drive.google.com/file/d/1A5xLXKkLZ49W_xxjS8b4wPm4SPa6yqa_/view?usp=sharing)
