from torch import nn
from transformers import BertModel


class FakeNewsClassifier(nn.Module):

    def __init__(self, n_classes, model_name):
        super(FakeNewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Note: We are ignoring the `token_type_ids` because they are not necessary for sequence classification
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Correct attribute for pooled output
        output = self.drop(pooled_output)
        return self.out(output)

