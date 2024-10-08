import torch

# Function to save the model
def save_model(model, epoch, path):
    torch.save(model.state_dict(), f'{path}/model_epoch_{epoch}.pth')

# Function to load a saved model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

# Function to display training progress (optional)
def display_training_progress(epoch, total_epochs, loss):
    print(f"Epoch [{epoch}/{total_epochs}], Loss: {loss:.6f}")

# Early stopping based on loss threshold (for overfitting prevention)
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

    def should_stop(self):
        return self.early_stop
