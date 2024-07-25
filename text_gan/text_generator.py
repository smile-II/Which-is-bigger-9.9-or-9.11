import torch
from transformers import BertTokenizer
from models.generator import Generator
from utils.helpers import load_model

def generate_samples(generator, tokenizer, num_samples=10, max_length=128):
    generator.eval()
    samples = []

    for _ in range(num_samples):
        input_text = "Random text"
        inputs = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        
        with torch.no_grad():
            logits = generator(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        generated_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
        samples.append(generated_text)
    
    return samples

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generator = Generator().cuda()
    
    # 加载保存的生成器模型
    generator = load_model(generator, 'text_gan/output/generator_epoch_1.pth')

    # 生成样本文本
    samples = generate_samples(generator, tokenizer, num_samples=10)

    # 打印生成的样本文本
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample}")

if __name__ == '__main__':
    main()
