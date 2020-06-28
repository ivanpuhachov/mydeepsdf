import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('data/chair.npy', 'rb') as f:
    features = torch.from_numpy(np.load(f))
    labels = torch.from_numpy(np.load(f))

dataset = TensorDataset(features, labels)
trainset, valset, testset = random_split(dataset, [200000, 10000, 40000])

train_loader = DataLoader(
    trainset,
    shuffle=True,
    batch_size=5000)
# TODO: paper uses batchsize = 16k, why?


def test_overfitting(mymodel, dataloader, lossfunction, learning_rate=1e-4, n_iters=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(mymodel)
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
        print(loss.item())


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


class Net(nn.Module):
    # TODO: weight normalization
    # TODO: dropouts
    def __init__(self, layer_dims):
        super(Net, self).__init__()
        input_dim = 3
        output_dim = 1
        layer_dims.append(output_dim)
        layer_dims.insert(0, input_dim)
        self.num_layers = len(layer_dims)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        # TODO: init_weights
        # self.init_weights()

    def forward(self, x):
        if x.is_cuda:
            device = x.get_device()
        else:
            device = torch.device("cpu")
        x.to(device)
        for layer in range(self.num_layers - 2):
            x = self.layers[layer](x)
            x = nn.functional.relu(x)
        x = self.layers[-1](x)
        x = torch.tanh(x)
        return x


model = Net([512, 512]).to(device)


def deepsdfloss(outputs, targets):
    # TODO: move that delta somewhere to clean up the code
    delta = 0.1
    # TODO: investigate nn.MSELoss()(torch.clamp(...) -...)
    return torch.mean(torch.abs(
        torch.clamp(outputs, min=-delta, max=delta) - torch.clamp(targets, min=-delta, max=delta)
    ))


# loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = deepsdfloss

# test_overfitting(model, train_loader, loss_fn)
test_training(model, train_loader, loss_fn)

# TODO: validation
# TODO: visualization
