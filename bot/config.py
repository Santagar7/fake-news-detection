import os

TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

MODEL_PATH = os.getenv('MODEL_PATH', 'path_to_your_model_directory/')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO')

DEFAULT_REPLY = "Sorry, I didn't understand that command."
