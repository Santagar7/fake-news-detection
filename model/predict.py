import torch
import torch.nn.functional as F
from data.preprocessor import preprocess
from transformers import BertTokenizer
from model.FakeNewsClassifier import FakeNewsClassifier

# Load the model
PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = FakeNewsClassifier(2, PRE_TRAINED_MODEL_NAME)


def load_model(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model


# Ensure the model is loaded and ready
model = load_model('model/bert/best_model_state.bin')


def make_prediction(text):
    inputs = preprocess(text, max_len=32)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        # Getting outputs directly from the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Applying softmax to convert to probabilities
        probabilities = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    return prediction.item(), confidence.item()

