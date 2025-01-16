import os
from telegram import Update
from telegram.ext import Updater, MessageHandler, filters, CallbackContext, ApplicationBuilder, CommandHandler
import nest_asyncio
import io
from pydub import AudioSegment
import numpy as np
import logging

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели Whisper
model = ("/content/drive/My Drive/Whisper_medium")

# Обработчик голосовых сообщений
async def handle_voice(update: Update, context: CallbackContext):
    try:
        # Получаем голосовое сообщение
        voice = update.message.voice
        file = await voice.get_file()

        # Скачиваем файл в виде байтов
        audio_bytes = await file.download_as_bytearray()

        # Преобразуем байты в аудиопоток
        audio_stream = io.BytesIO(audio_bytes)

        # Конвертируем .ogg в .wav с помощью pydub
        audio = AudioSegment.from_file(audio_stream, format="ogg")
        wav_stream = io.BytesIO()
        audio.export(wav_stream, format="wav")
        wav_stream.seek(0)  # Перемещаем указатель в начало потока

        # Сохраняем временный файл (опционально, для отладки)
        with open("temp_audio.wav", "wb") as f:
            f.write(wav_stream.getbuffer())

        # Распознаем текст с помощью Whisper
        result = model.transcribe("temp_audio.wav")
        text = result["text"]

        # Если текст пустой, отправляем сообщение об ошибке
        if not text:
            await update.message.reply_text("Не удалось распознать текст.")
        else:
            await update.message.reply_text(text)

    except Exception as e:
        logger.error(f"Ошибка при обработке голосового сообщения: {e}")
        await update.message.reply_text("Произошла ошибка при обработке голосового сообщения.")


# Обработчик команд /stop и /start
async def start_bot(update: Update, context: CallbackContext):
    await update.message.reply_text("Пожалуйста, пришлите голосовое сообщение")



async def stop_bot(update: Update, context: CallbackContext):
    await update.message.reply_text("Останавливаю бота...")
    await context.application.stop()
