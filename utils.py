import torch

# Function to save the model (not needed for inference but left for completeness)
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Function to load a saved model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
