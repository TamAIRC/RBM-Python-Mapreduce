import torch
import torch.nn as nn
import torch.optim as optim
from config.config import (
    DEVICE,
    LINK_DOWNLOAD_SAVE,
)
from RBM import RBM
from prepare_data import create_dataloader, download_mnist_data, split_dataset_by_labels


def pretrain_rbm(train_loader, visible_units, hidden_units, epochs=5, k=1, lr=0.01):
    rbm = RBM(visible_units, hidden_units).to(DEVICE)
    rbm.fit(train_loader, lr=lr, epochs=epochs, k=k)
    return rbm


class FineTunedAutoencoder(nn.Module):
    def __init__(self, rbms, hidden_layers):
        super(FineTunedAutoencoder, self).__init__()
        self.rbms = rbms
        self.encoder = nn.Sequential()
        for i, rbm in enumerate(self.rbms):
            self.encoder.add_module(f"rbm_{i}", rbm)

        self.mlp = nn.Sequential(nn.Linear(hidden_layers[-1], 10), nn.LogSoftmax(dim=1))

    def forward(self, x):
        for rbm in self.rbms:
            _, x = rbm.sample_h(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


def train_autoencoder(autoencoder, train_loader, valid_loader, epochs=5, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    autoencoder.to(DEVICE)

    for epoch in range(epochs):
        autoencoder.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = autoencoder(data.view(-1, 784))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)

        autoencoder.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = autoencoder(data.view(-1, 784))
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)

        valid_loss /= len(valid_loader.dataset)
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}"
        )


def main():
    # Download MNIST data
    train_dataset = download_mnist_data(LINK_DOWNLOAD_SAVE)

    # Split dataset
    train_set, valid_set, test_set = split_dataset_by_labels(train_dataset)

    # Create DataLoaders
    train_loader = create_dataloader(train_set)
    valid_loader = create_dataloader(valid_set)
    test_loader = create_dataloader(test_set)

    # Pre-train RBMs
    visible_units = 784
    hidden_units_list = [256, 128]
    rbms = []
    for hidden_units in hidden_units_list:
        rbm = pretrain_rbm(
            train_loader, visible_units, hidden_units, epochs=5, k=1, lr=0.01
        )
        rbms.append(rbm)
        visible_units = hidden_units

    # Fine-tune the model
    autoencoder = FineTunedAutoencoder(rbms, hidden_units_list)
    train_autoencoder(autoencoder, train_loader, valid_loader, epochs=5, lr=0.01)

    # Evaluate on test set
    autoencoder.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = autoencoder(data.view(-1, 784))
            test_loss += nn.functional.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
