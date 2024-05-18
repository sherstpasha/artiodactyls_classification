# Базовый образ с Python и поддержкой CUDA
FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка необходимых библиотек
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# Копирование исходного кода в контейнер
COPY . /app
WORKDIR /app

# Открытие консоли
CMD ["bash"]
