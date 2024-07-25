import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Generator(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(Generator, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.output_linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.output_linear(outputs.last_hidden_state)
        return logits

    def generate_from_latent(self, z, length=128):
        # 将潜在向量重塑为 (batch_size, length, hidden_size)
        batch_size = z.size(0)
        hidden_size = self.bert.config.hidden_size
        z = z.view(batch_size, length, hidden_size).to('cuda')

        # 使用BERT的嵌入层，将z直接作为嵌入输入BERT模型
        inputs_embeds = z
        token_type_ids = torch.zeros((batch_size, length), dtype=torch.long).cuda()
        attention_mask = torch.ones((batch_size, length), dtype=torch.long).cuda()

        # 使用BERT模型进行前向传播
        outputs = self.bert(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        logits = self.output_linear(outputs.last_hidden_state)

        # 从logits中选择概率最大的单词作为预测
        predictions = torch.argmax(logits, dim=-1)

        generated_texts = []
        for seq in predictions:
            tokens = [token_id.item() for token_id in seq]
            generated_texts.append(self.tokenizer.decode(tokens, skip_special_tokens=True))

        return generated_texts

# 示例用法
if __name__ == '__main__':
    generator = Generator().to("cuda")
    batch_size = 4
    length = 20
    hidden_size = generator.bert.config.hidden_size
    z = torch.randn(batch_size, length * hidden_size).to("cuda")

    generated_texts = generator.generate_from_latent(z, length=20)
    print(f"Generated texts: {generated_texts}")
