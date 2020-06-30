import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from models import SingleShapeSDF, deepsdfloss
import matplotlib.pyplot as plt
import meshplot
meshplot.offline()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('data/chair.npy', 'rb') as f:
    features = torch.from_numpy(np.load(f))
    labels = torch.from_numpy(np.load(f))

dataset = TensorDataset(features, labels)
trainset, valset, testset = random_split(dataset, [250000, 50000, 200000])

train_loader = DataLoader(
    trainset,
    shuffle=True,
    batch_size=5000)
# TODO: paper uses batchsize = 16k, why?

validation_loader = DataLoader(
    valset,
    shuffle=False,
    batch_size=5000
)

test_loader = DataLoader(
    testset, shuffle=False, batch_size=10000
)


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


def test_training(mymodel, dataloader, valloader, lossfunction, learning_rate=1e-4, n_epochs=10):
    print(mymodel)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    train_history = []
    validation_history = []
    for epoch in range(n_epochs):
        mymodel.train(True)
        print(f"\nEpoch {epoch}")
        running_loss = 0
        total_loss = 0
        for i, data in enumerate(dataloader, 0):
            x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
            y_pred = mymodel(x)
            loss = lossfunction(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 20 == 9:
                print(i, running_loss)
                running_loss = 0

        train_history.append(total_loss/20) # /20 to normalize and compare with validation set
        mymodel.train(False)
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valloader):
                x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
                y_pred = mymodel.forward(x)
                loss = lossfunction(y_pred, y)
                total_loss += loss.item()
        validation_history.append(total_loss)
    plt.plot(range(n_epochs), train_history, label='train')
    plt.plot(range(n_epochs), validation_history, label='val')


model = SingleShapeSDF([512, 512]).to(device)



# loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = deepsdfloss
n_epochs = 5
# test_overfitting(model, train_loader, loss_fn)
test_training(model, train_loader, validation_loader, loss_fn, n_epochs=n_epochs)

model.train(False)
test_loss = 0
testpoints = []
testsdf = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
        y_pred = model.forward(x)
        loss = loss_fn(y_pred, y)
        test_loss += loss.item()
    data = next(iter(test_loader))
    x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
    y_pred = model.forward(x)
    meshplot.plot(x.cpu().numpy(), c=y_pred.cpu().numpy(), shading={"point_size": 0.2}, filename="debug/predicted.html")
    meshplot.plot(x.cpu().numpy(), c=y.cpu().numpy(), shading={"point_size": 0.2}, filename="debug/target.html")
print(f"TEST LOSS: {test_loss/4}")
plt.axhline(y=test_loss/4, xmin=0, xmax=n_epochs-1, color='red', label='final test')
plt.legend()
plt.title("Loss")
plt.show()

# TODO: validation with another metric (not deepsdf loss)
# TODO: what metric do they use in the paper?
# TODO: visualization with marching cubes
