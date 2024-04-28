import torch
from data import download_and_prepare_data, split_data
from model import FakeNewsClassifier
from train import train
from utils import create_data_loader
from evaluate import eval_model, get_predictions, show_confusion_matrix
from config import DEVICE, PRE_TRAINED_MODEL_NAME, MAX_LEN, BATCH_SIZE
from transformers import BertTokenizer

df = download_and_prepare_data()
df_train, df_val, df_test = split_data(df)

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

model = FakeNewsClassifier(2)
model = model.to(DEVICE)

# Train the model
train(model, train_data_loader, val_data_loader, df_train, df_val)

# Evaluate the model
test_acc, test_loss = eval_model(model, test_data_loader, torch.nn.CrossEntropyLoss().to(DEVICE), DEVICE, len(df_test))
print(f'Test Accuracy: {test_acc} Test Loss: {test_loss}')

# Get predictions and show confusion matrix
y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader, DEVICE)
show_confusion_matrix(y_test, y_pred, ['Fake', 'Real'])
