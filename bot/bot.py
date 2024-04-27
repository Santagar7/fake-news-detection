from telegram.ext import CommandHandler, MessageHandler, filters, Application
from bot.config import TELEGRAM_TOKEN  # Ensure you have a config.py with TELEGRAM_TOKEN defined
from bot.handlers import start, help_command, echo


def run() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    application.run_polling()