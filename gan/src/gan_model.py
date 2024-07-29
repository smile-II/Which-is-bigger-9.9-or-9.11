import torch
from torch import nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig

class LoraAdapter(nn.Module):
    def __init__(self, base_model, peft_config):
        super(LoraAdapter, self).__init__()
        self.base_model = get_peft_model(base_model, peft_config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        return outputs.hidden_states[-1]  # 返回最后一层的 hidden states

class Generator(nn.Module):
    def __init__(self, lora_adapter):
        super(Generator, self).__init__()
        self.lora_adapter = lora_adapter

    def forward(self, input_ids, attention_mask=None):
        return self.lora_adapter(input_ids, attention_mask)

# class Discriminator(nn.Module):
#     def __init__(self, lora_adapter):
#         super(Discriminator, self).__init__()
#         self.lora_adapter = lora_adapter
#         self.classifier = nn.Linear(lora_adapter.base_model.config.padded_vocab_size, 1).half()


#     def forward(self, input_ids, attention_mask=None):
#         outputs = self.lora_adapter(input_ids, attention_mask).logits
#         print(f"outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
#         print(f"classifier weight shape: {self.classifier.weight.shape}, dtype: {self.classifier.weight.dtype}")
#         logits = self.classifier(outputs[:, 0, :])  # 使用CLS标记的表示
#         print(f"logits shape: {logits.shape}, dtype: {logits.dtype}")
#         return torch.sigmoid(logits)
class Discriminator(nn.Module):
    def __init__(self, lora_adapter):
        super(Discriminator, self).__init__()
        self.lora_adapter = lora_adapter
        self.classifier = nn.Linear(lora_adapter.base_model.config.hidden_size, 1).half()

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.lora_adapter(input_ids, attention_mask)
        print(f"hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        print(f"classifier weight shape: {self.classifier.weight.shape}, dtype: {self.classifier.weight.dtype}")
        logits = self.classifier(hidden_states[:, 0, :])  # 使用CLS标记的表示进行分类
        print(f"logits shape: {logits.shape}, dtype: {logits.dtype}")
        return logits


def initialize_models(model_name):
    base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1)
    lora_adapter = LoraAdapter(base_model, peft_config)

    generator = Generator(lora_adapter)
    discriminator = Discriminator(lora_adapter)

    return generator, discriminator
