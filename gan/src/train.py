import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from gan_model import initialize_models

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, prompt, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.process_data(self.texts[idx])
        return input_ids, attention_mask

    def process_data(self, data):
        # Add prompt to the data
        data_with_prompt = f"{self.prompt} {data}"
        encoded = self.tokenizer(data_with_prompt, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return encoded.input_ids.squeeze(), encoded.attention_mask.squeeze()

def train_gan(generator, discriminator, dataloader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for real_input_ids, real_attention_mask in dataloader:
            real_input_ids, real_attention_mask = real_input_ids.to(device).long(), real_attention_mask.to(device).long()

            # Prepare real labels and fake labels
            real_labels = torch.ones(real_input_ids.size(0), 1).to(device).half()
            fake_labels = torch.zeros(real_input_ids.size(0), 1).to(device).half()

            # Train Discriminator
            optimizer_d.zero_grad()
            real_outputs = discriminator(input_ids=real_input_ids, attention_mask=real_attention_mask)
            print(f"real_outputs shape: {real_outputs.shape}, dtype: {real_outputs.dtype}")
            real_loss = criterion(real_outputs, real_labels)
            real_loss.backward()

            fake_input_ids = generator(input_ids=real_input_ids, attention_mask=real_attention_mask)
            fake_hidden_states = fake_input_ids.last_hidden_state.detach()
            print(f"fake_hidden_states shape: {fake_hidden_states.shape}")

            fake_outputs = discriminator(input_ids=fake_hidden_states, attention_mask=real_attention_mask)
            print(f"fake_outputs shape: {fake_outputs.shape}, dtype: {fake_outputs.dtype}")
            fake_loss = criterion(fake_outputs, fake_labels)
            fake_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_input_ids = generator(input_ids=real_input_ids, attention_mask=real_attention_mask)
            fake_hidden_states = fake_input_ids.last_hidden_state
            fake_outputs = discriminator(input_ids=fake_hidden_states, attention_mask=real_attention_mask)
            gen_loss = criterion(fake_outputs, real_labels)
            gen_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch+1}/{epochs}] | D Loss: {real_loss.item()+fake_loss.item():.4f} | G Loss: {gen_loss.item():.4f}')

if __name__ == '__main__':
    model_name = '/home/home/txs/sjl/model/glm3'  # 使用一个小的预训练模型来快速测试
    generator, discriminator = initialize_models(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = "Which of these two numbers is larger? A: {} B: {}. \n###You must answer only with the single letter 'A' or 'B'. \n###You can only output a letter. \n###You output letter is:"

    # 加载IMDb数据集作为样本数据
    dataset = load_dataset('imdb', split='train[:1%]')  # 使用1%的训练数据来快速测试
    texts = dataset['text']
    
    text_dataset = TextDataset(texts, tokenizer, prompt)
    dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True)
    
    train_gan(generator, discriminator, dataloader)
