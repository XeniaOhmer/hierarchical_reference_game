import torch
import torch.nn as nn
import torch.nn.functional as F


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features, n_intentions):
        super(Sender, self).__init__()
        self.n_features = n_features
        self.fc1 = nn.Linear(n_features, n_hidden)      # embed input features
        self.fc2 = nn.Linear(n_intentions, n_hidden)      # embed intentions
        self.fc3 = nn.Linear(2 * n_hidden, n_hidden)    # linear layer to merge embeddings

    def forward(self, x, aux_input=None):
        features = x[:, 0:self.n_features]
        intentions = x[:, self.n_features:]
        feature_embedding = F.relu(self.fc1(features))
        intention_embedding = F.relu(self.fc2(intentions))
        joint_embedding = torch.cat([feature_embedding, intention_embedding], dim=1)
        return self.fc3(joint_embedding).tanh()


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)      # embed input features

    def forward(self, x, features, aux_input=None):
        feature_embeddings = self.fc1(features).tanh()
        energies = torch.matmul(feature_embeddings, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()
