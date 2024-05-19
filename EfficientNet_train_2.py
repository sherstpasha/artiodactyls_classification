import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
import argparse
import timm
import torch
from torch import nn, optim


ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataLoaderWrapper:
    def __init__(self, class_dirs, batch_size=64, image_size=(240, 240)):
        self.class_dirs = class_dirs
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        self.train_loader = None
        self.valid_loader = None
        self.num_classes = 3  # Кабарга, Косуля, Олень

    def load_valid_images(self, root):
        valid_samples = []
        dataset = datasets.ImageFolder(root=root, transform=self.transform)
        for sample in dataset.samples:
            try:
                with Image.open(sample[0]) as img:
                    img.verify()  # Проверка изображения
                    img.close()
                valid_samples.append(sample)
            except (IOError, OSError, ValueError) as e:
                print(f"Ошибка при загрузке изображения {sample[0]}: {e}")
        dataset.samples = valid_samples
        return dataset

    def setup_loaders(self):
        combined_train_samples = []
        combined_valid_samples = []

        # Обработка каждой директории класса
        for class_idx, class_dir in enumerate(self.class_dirs):
            train_dir = os.path.join(class_dir, 'train')
            valid_dir = os.path.join(class_dir, 'valid')

            train_dataset = self.load_valid_images(train_dir)
            valid_dataset = self.load_valid_images(valid_dir)

            combined_train_samples.extend(train_dataset.samples)
            combined_valid_samples.extend(valid_dataset.samples)

        # Создание объединенных датасетов
        combined_train_dataset = datasets.DatasetFolder(
            root='train', loader=self.loader, extensions=('jpg', 'jpeg', 'png'),
            transform=self.transform
        )
        combined_train_dataset.samples = combined_train_samples

        combined_valid_dataset = datasets.DatasetFolder(
            root='valid', loader=self.loader, extensions=('jpg', 'jpeg', 'png'),
            transform=self.transform
        )
        combined_valid_dataset.samples = combined_valid_samples

        self.train_loader = DataLoader(combined_train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(combined_valid_dataset, batch_size=self.batch_size, shuffle=False)

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def get_train_loader(self):
        if self.train_loader is None:
            self.setup_loaders()
        return self.train_loader

    def get_valid_loader(self):
        if self.valid_loader is None:
            self.setup_loaders()
        return self.valid_loader

    def get_count_classes(self):
        return self.num_classes


def train_and_validate(class_dirs, save_path, num_epochs, device):
    data_loader_wrapper = DataLoaderWrapper(class_dirs)

    train_loader = data_loader_wrapper.get_train_loader()
    val_loader = data_loader_wrapper.get_valid_loader()

    model_name = 'efficientnet_b1'
    model = timm.create_model(model_name, pretrained=True, num_classes=data_loader_wrapper.get_count_classes())

    # Определение устройства
    model.to(device)

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    print('Start training')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}')

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Val Accuracy: {val_accuracy}'")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {best_val_loss}")

    print('End training')
    return model


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

    return epoch_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate model.")
    parser.add_argument("kabarga_dir", type=str, help="Path to the Kabarga images directory.")
    parser.add_argument("kosulya_dir", type=str, help="Path to the Kosulya images directory.")
    parser.add_argument("olen_dir", type=str, help="Path to the Olen images directory.")
    parser.add_argument("num_epochs", type=int, help="Number of epochs for training.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to run the model on.")

    args = parser.parse_args()

    class_dirs = [args.kabarga_dir, args.kosulya_dir, args.olen_dir]

    # Обучение и валидация модели и сохранение
    trained_model = train_and_validate(class_dirs, save_path="best_model_efficientnet.pth",
                                       num_epochs=args.num_epochs, device=args.device)
