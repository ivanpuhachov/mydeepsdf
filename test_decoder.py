import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

class SDFDecoder(nn.Module):
    def __init__(self, layer_dims, latent_size=256):
        super(SDFDecoder, self).__init__()
        self.latent_size = latent_size
        self.latent_vector = torch.nn.Parameter(torch.randn([1, latent_size], dtype=torch.float32))
        self.layer_dims = layer_dims
        self.layer_dims.append(1)
        self.layer_dims.insert(0, 3+latent_size)
        self.num_layers = len(self.layer_dims)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

    def forward(self, xyz):
        batchsize = xyz.shape[0]
        inputs = torch.cat([self.latent_vector.repeat(batchsize,1), xyz], dim=1)
        x = inputs
        # TODO: skip-connection
        # TODO: weight norm
        # TODO: dropout
        for layer in range(self.num_layers - 2):
            x = self.layers[layer](x)
            x = nn.functional.relu(x)
        x = self.layers[-1](x)
        x = torch.tanh(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SDFDecoder([512, 512], latent_size=7).to(device)
print(model.latent_vector)

with open('data/chair.npy', 'rb') as f:
    features = torch.from_numpy(np.load(f))
    labels = torch.from_numpy(np.load(f))

dataset = TensorDataset(features, labels)
trainset, valset, testset = random_split(dataset, [200000, 10000, 40000])

train_loader = DataLoader(
    trainset,
    shuffle=True,
    batch_size=5000)

# TODO: deepSDF loss function
loss_fn = torch.nn.MSELoss(reduction='sum')


def test_overfitting(mymodel, dataloader, lossfunction, learning_rate=1e-4, n_iters=30):
    print(mymodel)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterat = iter(dataloader)
    d1, l1 = next(iterat)
    d1 = d1.to(device)
    l1 = l1.unsqueeze(1).to(device)
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    for i in range(n_iters):
        optimizer.zero_grad()
        o1 = mymodel.forward(d1)
        loss = lossfunction(o1, l1)
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            print(loss.item())
            print(model.latent_vector.sum().item())
            print()


def test_training(mymodel, dataloader, lossfunction, learning_rate=1e-4, n_epochs=10):
    print(mymodel)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch}")
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
            y_pred = mymodel(x)
            loss = lossfunction(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(i, running_loss)
                running_loss = 0


# test_overfitting(model, train_loader, loss_fn, learning_rate=1e-4, n_iters=100) # huge learning rate to validate that latent_vector is updating
test_training(model, train_loader, loss_fn, n_epochs=100)
print(model.latent_vector)
