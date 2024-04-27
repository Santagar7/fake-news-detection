from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess(text, max_len=512):
    # Tokenize and preprocess text identically to how it was done during training
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding
