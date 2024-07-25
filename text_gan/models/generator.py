import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Generator(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', latent_size=100):
        super(Generator, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(latent_size, self.bert.config.hidden_size)
        self.output_linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.latent_size = latent_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.output_linear(outputs.last_hidden_state)
        return logits

    def generate_from_latent(self, z, max_length=128):
        # 将 z 线性变换到 BERT 的隐藏大小
        hidden_states = self.linear(z).unsqueeze(1)  # (batch_size, 1, hidden_size)
        sequence = torch.zeros((z.size(0), max_length), dtype=torch.long).cuda()

        for i in range(max_length):
            outputs = self.output_linear(hidden_states)
            probabilities = torch.softmax(outputs, dim=-1)
            predicted_token = torch.argmax(probabilities, dim=-1)
            sequence[:, i] = predicted_token.squeeze()
            if i < max_length - 1:
                hidden_states = self.linear(z).unsqueeze(1)  # 更新hidden_states，保持形状一致

        generated_texts = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in sequence]
        return generated_texts

    # def generate(self, input_text, max_length=128):
    #     inputs = self.tokenizer.encode_plus(
    #         input_text,
    #         add_special_tokens=True,
    #         max_length=max_length,
    #         padding='max_length',
    #         truncation=True,
    #         return_tensors='pt'
    #     )
    #     input_ids = inputs['input_ids'].to('cuda')
    #     attention_mask = inputs['attention_mask'].to('cuda')
        
    #     with torch.no_grad():
    #         logits = self.forward(input_ids, attention_mask)
    #         predictions = torch.argmax(logits, dim=-1)
        
    #     return self.tokenizer.decode(predictions[0], skip_special_tokens=True)
