# config.py
import os

# You can store the token directly here, but it's not recommended for production
# TELEGRAM_TOKEN = 'your_telegram_bot_token_here'

# Instead, it's better to keep sensitive information as environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Other configurations can be added here
MODEL_PATH = os.getenv('MODEL_PATH', 'path_to_your_model_directory/')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO')

# You can also define default values and other configuration-related data
DEFAULT_REPLY = "Sorry, I didn't understand that command."
