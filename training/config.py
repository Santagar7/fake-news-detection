import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
BATCH_SIZE = 8
MAX_LEN = 512
EPOCHS = 10
