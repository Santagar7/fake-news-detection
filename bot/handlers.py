from telegram import Update
from telegram.ext import CallbackContext
from model.predict import make_prediction
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


async def handle_message(update: Update, context: CallbackContext) -> None:
    text = update.message.text
    # Make a prediction using the BERT model
    prediction, confidence = make_prediction(text)
    # Format the prediction into a response
    response = f"This news is {'real' if prediction == 1 else 'fake'} with a confidence of {confidence:.2f}."
    await update.message.reply_text(response)
