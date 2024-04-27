from telegram import Update
from telegram.ext import CallbackContext
# from model.predict import make_prediction  # Adjust import based on your project structure
import logging

# Enable logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


async def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Hi! Send me some text and I will analyze it using BERT.')


async def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text('Just send me some text and I will do sentiment analysis on it!')


async def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    user_text = update.message.text
    # prediction = make_prediction(user_text)  # Function to handle predictions
    # update.message.reply_text(f'Prediction: {prediction}')
