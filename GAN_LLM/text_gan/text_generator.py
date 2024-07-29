import torch
from transformers import BertTokenizer
from models.generator import Generator
from utils.helpers import load_model
            
def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generator = Generator().cuda()
    
    # 加载保存的生成器模型
    generator = load_model(generator, 'text_gan/output/generator_epoch_10.pth')
    batch_size = 8
    max_length = 128
    latent_size = 100

    # 初始化生成器和判别器
    G = Generator().cuda()
    # 生成样本文本
    z = torch.randn(batch_size, latent_size).cuda()
    fake_texts = G.generate_from_latent(z, max_length=max_length)

    # 打印生成的样本文本
    for i, sample in enumerate(fake_texts):
        print(f"Sample {i+1}: {sample}")

if __name__ == '__main__':
    main()
