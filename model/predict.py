import torch
from torch import softmax

from model.bert_model import create_bert_model
from data.preprocessor import preprocess

model_path = 'model/bert/bert.pth'


def load_model():
    model = create_bert_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to inference mode
    return model


model = load_model()


def make_prediction(text):
    # Preprocess the text
    inputs = preprocess(text, max_len=512)  # Make sure this matches training preprocessing

    # Predict
    prediction, confidence = make_prediction(inputs, model)
    response = f"Prediction: {'Fake' if prediction == 0 else 'Real'} with confidence {confidence:.2f}"

    inputs = preprocess(text)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    # Convert predicted class and confidence to Python scalars
    predicted_class = predicted_class.item()
    confidence = confidence.item()

    return predicted_class, confidence

