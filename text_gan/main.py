from transformers import BertTokenizer
from models.generator import Generator

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generator = Generator().cuda()

    input_text = "Hello, my name is"
    generated_text = generator.generate(input_text)
    print(f"Generated text: {generated_text}")

if __name__ == '__main__':
    main()