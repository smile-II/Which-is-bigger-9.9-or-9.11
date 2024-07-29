import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class Generator(nn.Module):
    def __init__(self, gpt2_model_name='gpt2'):
        super(Generator, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        self.output_linear = nn.Linear(self.gpt2.config.hidden_size, self.gpt2.config.vocab_size)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        
        # 添加pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.output_linear(outputs.last_hidden_state)
        return logits

    def generate_from_latent(self, z, length=128):
        # 将潜在向量重塑为 (batch_size, length, hidden_size)
        batch_size = z.size(0)
        hidden_size = self.gpt2.config.hidden_size
        z = z.view(batch_size, length, hidden_size).to('cuda')

        print(f'z shape: {z.shape}')  # 打印 z 的形状

        # GPT-2 不直接支持 inputs_embeds，所以需要自定义 forward
        inputs_embeds = z
        attention_mask = torch.ones((batch_size, length), dtype=torch.long).to('cuda')

        print(f'inputs_embeds shape: {inputs_embeds.shape}')  # 打印 inputs_embeds 的形状
        print(f'attention_mask shape: {attention_mask.shape}')  # 打印 attention_mask 的形状

        # 使用GPT-2模型进行前向传播
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        
        logits = self.output_linear(outputs.last_hidden_state)

        print(f'logits shape: {logits.shape}')  # 打印 logits 的形状

        # 使用温度采样和随机性引入更多多样性
        temperature = 1.0
        logits = logits / temperature
        probabilities = torch.softmax(logits, dim=-1)

        # 从logits中选择概率最大的单词作为预测
        predictions = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), 1).view(batch_size, length)

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
    hidden_size = generator.gpt2.config.hidden_size
    z = torch.randn(batch_size, length * hidden_size).to("cuda")

    generated_texts = generator.generate_from_latent(z, length=20)
    print(f"Generated texts: {generated_texts}")
