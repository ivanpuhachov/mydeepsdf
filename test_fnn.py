import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from models import SingleShapeSDF, deepsdfloss
import matplotlib.pyplot as plt
import argparse
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import get_sdfgrid
import meshplot

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(mymodel)
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
    t_history = []
    v_history = []
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

        t_history.append(total_loss/20) # /20 to normalize and compare with validation set
        mymodel.train(False)
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valloader):
                x, y = data[0].to(device), data[1].unsqueeze(1).to(device)
                y_pred = mymodel.forward(x)
                loss = lossfunction(y_pred, y)
                total_loss += loss.item()
        v_history.append(total_loss)
    return t_history, v_history

def visualize_voxels(model, grid_res=20):
    outs = get_sdfgrid(model, grid_res)
    sdfs_ = (np.abs(outs) < 1 / grid_res) * 1.0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(sdfs_, edgecolor="k")
    plt.show()


def visualize_marchingcubes(model, grid_res=100):
    outs = get_sdfgrid(model, grid_res)
    verts, faces, normals, values = measure.marching_cubes(outs, 0)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(0, grid_res)
    ax.set_ylim(0, grid_res)
    ax.set_zlim(0, grid_res)

    plt.tight_layout()
    plt.show()

def plot_training_curve(train_history, validation_history, final_loss):
    n_epochs = len(train_history)
    plt.plot(range(n_epochs), train_history, label='train')
    plt.plot(range(n_epochs), validation_history, label='val')
    plt.axhline(y=final_loss/4, xmin=0, xmax=n_epochs-1, color='red', label='final test')
    plt.legend()
    plt.title("Loss")
    plt.show()


def main():
    meshplot.offline()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    # TODO: change default values
    parser.add_argument("-i", "--input", help="Path to .npy file. Default: 'data/chair.npy'",
                        default='data/chair.npy')
    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs. Default: 50", default=50)
    parser.add_argument("-b", "--batch", type=int, help="Batch size. Default: 5000", default=5000)
    parser.add_argument("-r", "--rate", type=float, help="learning rate. Default: 1e-4", default=1e-4)
    args = parser.parse_args()

    file_path = args.input
    n_epochs = args.epochs
    lr = args.rate
    bs = args.batch

    with open(file_path, 'rb') as f:
        xyz = np.load(f)
        dataset_size = xyz.shape[0]
        features = torch.from_numpy(xyz)
        labels = torch.from_numpy(np.load(f))

    dataset = TensorDataset(features, labels)
    trainset, valset, testset = random_split(dataset, [250000, 10000, 40000])

    train_loader = DataLoader(
        trainset,
        shuffle=True,
        batch_size=bs)

    validation_loader = DataLoader(
        valset,
        shuffle=False,
        batch_size=bs
    )

    test_loader = DataLoader(
        testset, shuffle=False, batch_size=bs
    )

    model = SingleShapeSDF([512, 512, 512]).to(device)

    # loss_fn = torch.nn.MSELoss(reduction='sum')
    loss_fn = deepsdfloss
    # test_overfitting(model, train_loader, loss_fn)
    train_history, validation_history = test_training(model, train_loader, validation_loader, loss_fn,
                                                      n_epochs=n_epochs, learning_rate=lr)

    model.eval()
    test_loss = 0
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





    plot_training_curve(train_history, validation_history, test_loss)

    # TODO: validation with another metric (not deepsdf loss)
    # TODO: what metric do they use in the paper?




    visualize_voxels(model, grid_res=20)
    visualize_marchingcubes(model, grid_res=100)


if __name__ == '__main__':
    main()