import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Discriminator(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(Discriminator, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)
