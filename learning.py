import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


model_name = "openai/whisper-medium"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
processor = WhisperProcessor.from_pretrained(model_name)

# Загрузка данных Mozilla Common Voice (русский язык)
dataset = ("/content/drive/My Drive/common_voice_20_0", "ru")

# Подготовка данных
def preprocess_function(batch):
    # Процессинг аудио
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    # Токенизация текста
    with processor.as_target_processor():
        labels = processor(batch["sentence"], return_tensors="pt", padding=True).input_ids
    inputs["labels"] = labels
    return inputs

# Применение функции к обучающей выборке
train_data = dataset["train"].map(preprocess_function, remove_columns=dataset["train"].column_names)
val_data = dataset["validation"].map(preprocess_function, remove_columns=dataset["validation"].column_names)

# Даталоадеры
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# Оптимизатор и параметры обучения
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# Функция для обучения модели
def train(model, train_loader, val_loader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Training Loss: {epoch_loss / len(train_loader)}")

        # Оценка на валидационной выборке
        evaluate(model, val_loader)

# Функция для оценки модели
def evaluate(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            val_loss += outputs.loss.item()
    print(f"Validation Loss: {val_loss / len(val_loader)}")
    model.train()

# Запуск обучения
train(model, train_loader, val_loader, optimizer, epochs=5)
