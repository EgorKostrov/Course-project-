from telegram.ext import Updater, MessageHandler, filters, CallbackContext, ApplicationBuilder, CommandHandler


# Запуск бота
def main():
    # Укажите токен вашего бота
    application = ApplicationBuilder().token("").build()

    # Регистрируем обработчик голосовых сообщений
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(CommandHandler("start", start_bot))
    application.add_handler(CommandHandler("stop", stop_bot))

    # Запускаем бота
    application.run_polling()

if __name__ == '__main__':
    main()
