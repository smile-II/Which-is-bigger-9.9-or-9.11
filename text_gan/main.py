from transformers import BertTokenizer
from models.generator import Generator

def main():
    # 初始化tokenizer和生成器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generator = Generator().cuda()

    input_text = "Hello, my name is"
    # 生成文本
    generated_text = generator.generate(input_text)
    print(f"Generated text: {generated_text}")

if __name__ == '__main__':
    main()
