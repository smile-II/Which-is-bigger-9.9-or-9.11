import torch
import torch.nn as nn
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator
from data.data_loader import get_data_loader

# 训练过程
def train():
    batch_size = 16
    max_length = 128
    hidden_size = 768  # BERT hidden size
    latent_size = 100
    num_epochs = 5
    learning_rate = 0.0002

    texts = ["Example text data for training."] * 100  # Replace with your dataset
    data_loader = get_data_loader(texts, batch_size, max_length)

    G = Generator().cuda()
    D = Discriminator().cuda()

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(D.parameters(), lr=learning_rate)
    optimizerG = optim.Adam(G.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (input_ids, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

            real_labels = torch.ones(batch_size, 1).cuda()
            fake_labels = torch.zeros(batch_size, 1).cuda()

            # 训练判别器
            real_outputs = D(input_ids, attention_mask)
            d_loss_real = criterion(real_outputs, real_labels)

            z = torch.randn(batch_size, latent_size).cuda()
            fake_texts = [G.generate("Random text") for _ in range(batch_size)]
            fake_inputs = G.tokenizer(fake_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            fake_ids = fake_inputs['input_ids'].cuda()
            fake_attention_mask = fake_inputs['attention_mask'].cuda()

            fake_outputs = D(fake_ids, fake_attention_mask)
            d_loss_fake = criterion(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # 训练生成器
            z = torch.randn(batch_size, latent_size).cuda()
            fake_texts = [G.generate("Random text") for _ in range(batch_size)]
            fake_inputs = G.tokenizer(fake_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            fake_ids = fake_inputs['input_ids'].cuda()
            fake_attention_mask = fake_inputs['attention_mask'].cuda()

            outputs = D(fake_ids, fake_attention_mask)
            g_loss = criterion(outputs, real_labels)

            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

if __name__ == '__main__':
    train()