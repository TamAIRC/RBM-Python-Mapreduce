import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE


class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units

        # Initialize weights
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))  # hidden layer bias
        self.v_bias = nn.Parameter(torch.zeros(visible_units))  # visible layer bias

    def sample_h(self, v):
        # Compute hidden layer activations and sample from the distribution
        h_prob = torch.sigmoid(nn.functional.linear(v, self.W, self.h_bias))
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        # Compute visible layer activations and sample from the distribution
        v_prob = torch.sigmoid(nn.functional.linear(h, self.W.t(), self.v_bias))
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        h_prob, h_sample = self.sample_h(v)
        v_prob, v_sample = self.sample_v(h_sample)
        return v_prob

    def contrastive_divergence(self, v, k=1):
        v0 = v
        for _ in range(k):
            h_prob, h_sample = self.sample_h(v)
            v_prob, v_sample = self.sample_v(h_sample)
            v = v_sample

        # Positive and negative phase
        positive_grad = torch.mm(v0.t(), h_prob)
        negative_grad = torch.mm(v_prob.t(), h_sample)

        self.W.grad = (positive_grad - negative_grad).t() / v0.size(0)
        self.v_bias.grad = torch.mean(v0 - v_prob, dim=0)
        self.h_bias.grad = torch.mean(h_prob - h_sample, dim=0)

    def fit(self, train_loader, lr=0.01, epochs=5, k=1):
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for data, _ in train_loader:
                v = data.view(-1, self.visible_units)
                v = v.to(DEVICE)
                self.contrastive_divergence(v, k)

                optimizer.step()
                epoch_loss += torch.sum((v - self.forward(v)) ** 2).item()

            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader.dataset):.4f}"
            )
