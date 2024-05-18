# artiodactyls_classification


сборка контейнера
docker build -t your_image_name .


запуск валидации для ResNet
python ResNet_eval.py C:/Users/user/Desktop/data/valid C:/Users/user/Desktop/sorted_predictions --model_path best_model_ResNet_20.pth --device cuda


запуск обучения для ResNet
python ResNet_train.py C:/Users/user/Desktop/data/train C:/Users/user/ack/artiodactyls_classification/best_model_ResNet_20.pth 10 --device cpu

запуск предсказания для ResNet
python ResNet_pred.py C:/Users/user/Desktop/test_valid C:/Users/user/Desktop/sorted_predictions --model_path best_model_ResNet_20.pth --device cpu


ссылка на ResNet https://drive.google.com/file/d/1TTkXIEqNbkzgpsFuLDUzvikLnzqJFay_/view?usp=sharing

ссылка на yolo5s https://drive.google.com/file/d/1A5xLXKkLZ49W_xxjS8b4wPm4SPa6yqa_/view?usp=sharing


