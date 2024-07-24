import torch

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model
