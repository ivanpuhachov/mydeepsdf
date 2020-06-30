import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from models import FamilyShapeDecoderSDF, deepsdfloss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FamilyShapeDecoderSDF(latent_size=7, family_size=10).to(device)

with open('data/chair.npy', 'rb') as f:
    features = torch.from_numpy(np.load(f))
    labels = torch.from_numpy(np.load(f))

dataset = TensorDataset(features, labels)
trainset, valset, testset = random_split(dataset, [250000, 50000, 200000])

train_loader = DataLoader(
    trainset,
    shuffle=True,
    batch_size=5000)


def test_overfitting(mymodel, dataloader, lossfunction, learning_rate=1e-4, n_iters=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterat = iter(dataloader)
    d1, l1 = next(iterat)
    d1 = d1.to(device)
    l1 = l1.unsqueeze(1).to(device)
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    for i in range(n_iters):
        optimizer.zero_grad()
        o1 = mymodel.forward(d1, family_id=0)
        loss = lossfunction(o1, l1)
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            print(loss.item())
            print(model.latent_vector.sum().item())
            print()


def test_training(mymodel, dataloader, lossfunction, learning_rate=1e-4, n_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch}")
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
            y_pred = mymodel.forward(x, family_id=0)
            loss = lossfunction(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(i, running_loss)
                running_loss = 0


print(model.latent_vector[0])
# test_overfitting(model, train_loader, deepsdfloss, learning_rate=1e-4, n_iters=100)
test_training(model, train_loader, deepsdfloss, n_epochs=20)
print(model.latent_vector[0])