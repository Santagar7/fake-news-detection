import torch


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    return model
