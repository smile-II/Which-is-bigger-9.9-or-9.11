import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Generator(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(Generator, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.last_hidden_state)
        return logits

    def generate(self, input_text, max_length=128):
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        return self.tokenizer.decode(predictions[0], skip_special_tokens=True)