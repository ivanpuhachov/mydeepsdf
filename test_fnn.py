import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

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


# TODO: model as class, hidden layers dims as init params
model = torch.nn.Sequential(
    torch.nn.Linear(3, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1),
    torch.nn.Tanh()
).to(device)

# TODO: deepsdf loss function
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(30):
    print(f"\nEpoch {epoch}")
    for i, data in enumerate(train_loader, 0):
        x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        if i % 10 == 9:
            print(i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# TODO: validation
# TODO: visualization
