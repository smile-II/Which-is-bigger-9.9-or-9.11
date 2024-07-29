import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision

from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_data_loader

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def train():
    batch_size = 64
    image_size = 28 * 28  # 784
    hidden_size = 256
    latent_size = 64
    num_epochs = 100
    learning_rate = 0.0002

    data_loader = get_data_loader(batch_size)

    G = Generator(latent_size, hidden_size, image_size).to('cuda')
    D = Discriminator(image_size, hidden_size, 1).to('cuda')

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(D.parameters(), lr=learning_rate)
    optimizerG = optim.Adam(G.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            # 展平图像为 (batch_size, 784)
            images = images.view(batch_size, -1)
            
            # 捕捉异常图像尺寸
            if images.size(1) != 784:
                print(f'Error at epoch {epoch+1}, step {i+1}: Image size after view: {images.size()}')
                continue  # 跳过异常图像
            
            images = images.to('cuda')
            real_labels = torch.ones(batch_size, 1).to('cuda')
            fake_labels = torch.zeros(batch_size, 1).to('cuda')

            # 训练判别器
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels) #判别器和真实标签做损失，学会判别真实标签
            real_score = outputs 

            z = torch.randn(batch_size, latent_size).to('cuda')
            print(z)
            fake_images = G(z) #生成器生成的图像
            outputs = D(fake_images) #生成假图像判别
            d_loss_fake = criterion(outputs, fake_labels) #判别成假标签
            fake_score = outputs

            d_loss = d_loss_real + d_loss_fake #增加loss 相加
            optimizerD.zero_grad() #清除梯度
            d_loss.backward() #计算梯度
            optimizerD.step() #更新参数

            # 训练生成器
            z = torch.randn(batch_size, latent_size).to('cuda')
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels) #计算生成器生成的图像被判别后和真标签的损失
            
            

            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                      f'D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')

    z = torch.randn(batch_size, latent_size).to('cuda')
    fake_images = G(z)
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    fake_images = denorm(fake_images)

    grid = torchvision.utils.make_grid(fake_images, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.cpu().permute(1, 2, 0).squeeze())
    plt.show()

if __name__ == '__main__':
    train()
