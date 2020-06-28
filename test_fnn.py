import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from models import SingleShapeSDF, deepsdfloss

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



model = SingleShapeSDF([512, 512]).to(device)



# loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = deepsdfloss

# test_overfitting(model, train_loader, loss_fn)
test_training(model, train_loader, loss_fn)

# TODO: validation
# TODO: visualization
