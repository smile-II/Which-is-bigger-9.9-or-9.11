import torch
from models.generator import Generator
from models.discriminator import Discriminator

def main():
    latent_size = 64
    hidden_size = 256
    image_size = 28 * 28
    
    G = Generator(latent_size, hidden_size, image_size).to('cuda')
    D = Discriminator(image_size, hidden_size, 1).to('cuda')

    print(G)
    print(D)

if __name__ == '__main__':
    main()