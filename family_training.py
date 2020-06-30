import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from models import FamilyShapeDecoderSDF, deepsdfloss
import argparse
from os import path, listdir
from models import FamilyShapeDecoderSDF
import random

parser = argparse.ArgumentParser()
# TODO: change default values
parser.add_argument("-i", "--input", help="Path to parent folder of obj. Default: 'data/airplanes/npy/'",
                    default='debug/temp/')
parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs. Default: 5", default=2)
parser.add_argument("-l", "--latent", type=int, help="Dimensionality of the latent space. Default: 256", default=7)
parser.add_argument("-b", "--batch", type=int, help="Batch size. Default: 5000", default=5000)
args = parser.parse_args()

folderpath = args.input
n_epochs = args.epochs
latent_size = args.latent
batch_size = args.batch

assert(path.exists(folderpath))

family_size = len(listdir(folderpath))
assert(family_size > 0)

family_data_train = dict()
family_data_test = dict()

for filename in listdir(folderpath):
    with open(folderpath+filename, 'rb') as f:
        features = torch.from_numpy(np.load(f))
        labels = torch.from_numpy(np.load(f))
    dataset = TensorDataset(features, labels)
    trainset, testset = random_split(dataset, [250000, 50000])
    train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(testset, shuffle=True, batch_size=batch_size)
    id_ = int(filename[:filename.find(".npy")])
    family_data_train[filename] = (id_, train_loader)
    family_data_test[filename] = (id_, test_loader)

model = FamilyShapeDecoderSDF(layer_dims=[512, 512], family_size=family_size, latent_size=latent_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(n_epochs):
    print(f"\n----------------\nEpoch {epoch}")
    # random order of shapes training
    for key, value in sorted(family_data_train.items(), key=lambda x: random.random()):
        id_, trainloader_ = value
        print(f"\nShape {id_}")
        running_loss = 0
        print(model.latent_vector[id_, :])
        for i, data in enumerate(trainloader_, 0):
            x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
            y_pred = model.forward(x, family_id=id_)
            loss = deepsdfloss(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 10 == 9:
            #     print(i, running_loss)
            #     running_loss = 0
        print(model.latent_vector[id_, :])


