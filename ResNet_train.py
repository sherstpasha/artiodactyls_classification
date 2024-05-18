import os
import shutil
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import timm
import torch.nn as nn
import torch.optim as optim
import cv2
import argparse

# Разрешить загрузку поврежденных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.image_folder = ImageFolder(root_dir, transform=self.transform)
        
    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        try:
            image, label = self.image_folder[idx]
            inputs = self.processor(image).unsqueeze(0)
            return {'pixel_values': inputs, 'labels': torch.tensor(label)}
        except (OSError, IOError) as e:
            print(f"Error loading image at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def transform(self, image):
        return image.convert("RGB")

def train_and_validate(data_dir, model_path, save_path, num_epochs, device):
    # Загрузка предобученной модели из timm
    model = timm.create_model('resnext50_32x4d.a3_in1k', pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Получение трансформаций модели (нормализация, изменение размера)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Создание датасета
    dataset = CustomImageDataset(data_dir, transforms)

    # Разделение данных на обучающую и валидационную выборки
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Настройка модели, критерия, оптимизатора и планировщика
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            inputs = batch['pixel_values'].squeeze(1).to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}")
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with validation loss: {best_val_loss}")

    return model

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['pixel_values'].squeeze(1).to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate model.")
    parser.add_argument("data_dir", type=str, help="Path to the input data directory.")
    parser.add_argument("model_path", type=str, help="Path to the saved model.")
    parser.add_argument("num_epochs", type=int, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the model on.")

    args = parser.parse_args()

    # Обучение и валидация модели и сохранение
    trained_model = train_and_validate(args.data_dir, args.model_path, save_path="best_model.pth", num_epochs=args.num_epochs, device=args.device)
