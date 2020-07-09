import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from models import FamilyShapeDecoderSDF, deepsdfloss, l1loss
import argparse
from os import path, listdir, mkdir
import random
from utils import get_torchgrid, get_balancedsampler
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from mesh_to_sdf import sample_sdf_near_surface
import trimesh


class FamilyShapeSDFWrapper:
    def __init__(self, family_size=50,
                 latent_size=10,
                 h_blocks=40,
                 batch_size=5000,
                 path_to_training_npyfolder='data/airplanes/npy/',
                 path_to_saves='log/'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.family_size = family_size
        self.latent_size = latent_size
        self.h_blocks = h_blocks
        self.batch_size = batch_size
        self.model = FamilyShapeDecoderSDF(family_size=family_size, latent_size=latent_size,
                                           h_blocks=h_blocks).to(self.device)
        self.path_to_training_npyfolder = path_to_training_npyfolder
        self.path_to_saves = path_to_saves
        assert (path.exists(self.path_to_training_npyfolder))
        assert (len(listdir(path_to_training_npyfolder)) <= family_size)
        self.family_data_train = dict()
        self.family_data_validation = dict()
        self.filename_to_id = dict()
        self.id_to_filename = dict()
        self.train_history = []
        self.validation_history = []
        self.get_loaders()

    def save(self):
        # TODO: prettify and use pickle
        data = dict()
        data['family_size'] = self.family_size
        data['latent_size'] = self.latent_size
        data['h_blocks'] = self.h_blocks
        data['batch_size'] = self.batch_size
        data['path_to_training_npyfolder'] = self.path_to_training_npyfolder
        data['path_to_saves'] = self.path_to_saves
        data['filename_to_id'] = self.filename_to_id
        data['id_to_filename'] = self.id_to_filename
        data['train_history'] = self.train_history
        data['validation_history'] = self.validation_history
        with open(self.path_to_saves+"data.json", "w") as jsonfile:
            json.dump(data, jsonfile)
        torch.save(self.model.state_dict(), self.path_to_saves + "model-parameters.pt")

    # def load(self, path_to_save_folder: str):
    #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     with open(path_to_save_folder + 'data.json') as jsonfile:
    #         data = json.load(jsonfile)
    #         self.path_to_training_npyfolder=data['path_to_training_npyfolder']
    #         self.batch_size = data['batch_size']
    #         self.h_blocks = data['h_blocks']
    #         self.latent_size = data['latent_size']
    #         self.path_to_saves = data['path_to_saves']
    #         self.model.load_state_dict(torch.load(path_to_save_folder + "model-parameters.pt"))
    #         self.get_loaders()
    #         new_filename_to_id = data['filename_to_id']
            # TODO: finish load

    def get_loaders(self):
        for id_, filename in enumerate(listdir(self.path_to_training_npyfolder)):
            # TODO: use torch.utils.data.WeightedRandomSampler
            with open(self.path_to_training_npyfolder + filename, 'rb') as f:
                features = np.load(f)
                labels = np.load(f)
            # balanced sampling
            sampler = get_balancedsampler(labels)
            dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
            trainset, validationset = random_split(dataset, [250000, 50000])
            train_loader = DataLoader(trainset, sampler=sampler, batch_size=self.batch_size, num_workers=8)
            validation_loader = DataLoader(validationset, shuffle=True, batch_size=self.batch_size, num_workers=8)
            self.filename_to_id[filename] = id_
            self.family_data_train[filename] = (id_, train_loader)
            self.family_data_validation[filename] = (id_, validation_loader)
            self.filename_to_id[id_] = filename

    def train(self, n_epochs=5, learning_rate=1e-4, debug=False, lossfunction=deepsdfloss):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if debug:
            print("\n Latent vector 0:")
            print(self.model.latent_vector[0])
            print("\n Last layer weights: ")
            print(self.model.block1[7].weight)
        val_best = 100
        for epoch in range(n_epochs):
            print(f"\n----------------\nEpoch {epoch}")
            self.model.train()
            total_loss = 0
            # random order of shapes training
            for key, value in sorted(self.family_data_train.items(), key=lambda xx: random.random()):
                id_, trainloader_ = value
                running_loss = 0
                regularizer = 0
                if debug:
                    print(self.model.latent_vector[id_, :])
                for i, data in enumerate(trainloader_, 0):
                    x, y = data[0].to(self.device), data[1].unsqueeze(1).to(self.device)
                    y_pred = self.model(x, family_id=id_)
                    regularizer = torch.norm(self.model.latent_vector[id_]) / 0.01
                    loss = lossfunction(y_pred, y) + regularizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() - regularizer.item()  # (so we add only lossfunction)
                running_loss += regularizer.item()
                total_loss += running_loss
                print(f"Shape {id_} - {self.filename_to_id[id_]}, loss: {running_loss}")
                if debug:
                    print(self.model.latent_vector[id_, :])
            self.train_history.append(total_loss)
            self.model.eval()
            val_loss = self.validate(lossfunction, debug)
            print(f"\nTotal validation loss: {val_loss}")
            self.validation_history.append(val_loss)
            if val_loss < val_best:
                print("save-best")
                val_best = val_loss
                torch.save(self.model.state_dict(), self.path_to_saves+"best-model-parameters.pt")
            self.plot_history()
        # load best
        self.model.load_state_dict(torch.load(self.path_to_saves+"best-model-parameters.pt"))

        if debug:
            print("\n Latent vector 0:")
            print(self.model.latent_vector[0])
            print("\n Last layer weights: ")
            print(self.model.block1[7].weight)

    def validate(self, lossfunction=l1loss, debug=False):
        total_loss = 0
        for key, value in self.family_data_validation.items():
            id_, validationloader_ = value
            latent = self.model.latent_vector[id_, :]
            loss = self.val_(validationloader_, latent, lossfunction)
            if debug:
                print(f"Shape {key} - {self.filename_to_id[key]} loss: {loss}")
            total_loss += loss
        if debug:
            print(f"Total loss: {total_loss}")
        return total_loss

    def val_(self, dataloader, latentvector, lossfunction=l1loss):
        assert latentvector.shape[0] == self.latent_size
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                x, y = data[0].to(self.device), data[1].unsqueeze(1).to(self.device)
                y_pred = self.model.forward_customlatent(x, latentvector)
                loss = lossfunction(y_pred, y)
                val_loss += loss.item()
        return val_loss

    def visualize_id_voxels(self, latent_id=0, grid_res=20):
        latent = self.model.latent_vector[latent_id]
        self.visualize_latent_voxels(latent_vector=latent, grid_res=grid_res)

    def visualize_latent_voxels(self, latent_vector, grid_res=20):
        outs = self.evaluate_on_grid(latent_vector=latent_vector, grid_res=grid_res)
        sdfs_ = (np.abs(outs) < 1 / grid_res) * 1.0
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(sdfs_, edgecolor="k")
        plt.show()

    def visualize_id_marchingcubes(self, latent_id=0, grid_res=50):
        latent = self.model.latent_vector[latent_id]
        self.visualize_latent_marchingcubes(latent_vector=latent, grid_res=grid_res)

    def visualize_latent_marchingcubes(self, latent_vector, grid_res=50):
        try:
            outs = self.evaluate_on_grid(latent_vector=latent_vector, grid_res=grid_res)
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
        # TODO: write error
        except:
            print("\nProblem with Marching Cubes, visualizing voxels instead")
            self.visualize_latent_voxels(latent_vector)

    def evaluate_on_grid(self, latent_vector, grid_res=20):
        grid = get_torchgrid(grid_res, self.device)
        self.model.eval()
        with torch.no_grad():
            outs = self.model.forward_customlatent(grid, latent=latent_vector).cpu().reshape(
                shape=(grid_res, grid_res, grid_res)).numpy()
        return outs

    def get_id_by_filename(self, filename):
        if filename in self.filename_to_id:
            return self.filename_to_id[filename]
        else:
            print("filename not valid")
            return 0

    def get_latent_by_filename(self, filename):
        id_ = self.get_id_by_filename(filename)
        return self.model.latent_vector[id_, :]

    def plot_history(self, filename="training.png"):
        n_epoch = len(self.train_history)
        plt.figure()
        plt.plot(range(n_epoch), self.train_history, label='train')
        plt.plot(range(n_epoch), np.array(self.validation_history)*5, label='val')  # TODO: remove this constant
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(self.path_to_saves + filename)
        plt.close()

    def fit_latent_for_mesh(self, filepath, points_sampled=300000, n_epochs=20,
                            learning_rate=1e-3, loss=deepsdfloss, debug=False):
        print("Loading mesh")
        mesh = trimesh.load(filepath)
        print("Sampling points")
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=points_sampled)
        dataset = TensorDataset(torch.from_numpy(points), torch.from_numpy(sdf))
        dataloader = DataLoader(dataset, shuffle=True, batch_size=5000)
        print("Optimizing latent")
        return self.fit_latent_for_dataloader(dataloader, n_epochs, learning_rate, loss, debug)

    def fit_latent_for_dataloader(self, dataloader, n_epochs=20, learning_rate=1e-3, loss=deepsdfloss, debug=False):
        self.model.eval()
        lat = torch.autograd.Variable(torch.randn([self.latent_size], dtype=torch.float32)).to(self.device)
        # self.visualize_latent_voxels(latent_vector=lat)
        if debug:
            print(lat)
        lat.requires_grad = True
        momentum = torch.zeros(self.latent_size).to(self.device)
        for epoch in range(n_epochs):
            running_loss = 0
            for i, data in enumerate(dataloader, 0):
                x, y = data[0].to(self.device), data[1].unsqueeze(1).to(self.device)
                self.model.zero_grad()
                y_pred = self.model.forward_customlatent(x, latent=lat)
                c_loss = loss(y_pred, y)
                running_loss += c_loss.item()
                c_loss.backward()
                average_grad = np.abs(lat.grad.data.cpu().numpy()).mean()
                normalized_lr = learning_rate / average_grad
                momentum = 0.1 * momentum + lat.grad.data
                lat.data = lat.data - normalized_lr * lat.grad.data
                lat.grad.data.zero_()
            print(f"Fit epoch {epoch}, loss {running_loss}")
        if debug:
            print(lat)
        # self.visualize_latent_voxels(latent_vector=lat)
        return lat


def get_parser():
    parser = argparse.ArgumentParser()
    # TODO: change default values
    parser.add_argument("-i", "--input", help="Path to parent folder of npy. Default: 'data/airplanes/npy/'",
                        default='data/airplanes/npy/')
    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs. Default: 5", default=5)
    parser.add_argument("-l", "--latent", type=int, help="Dimensionality of the latent space. Default: 256", default=256)
    parser.add_argument("-b", "--batch", type=int, help="Batch size. Default: 16384", default=16384)
    parser.add_argument("-g", "--height", type=int, help="Number of neurons in hidden units. Default: 512", default=512)
    parser.add_argument("-d", "--debug", help="Print debugging info", action='store_true')
    parser.add_argument("-r", "--rate", type=float, help="Learning rate. Default: 1e-4", default=1e-4)
    return parser


def load_wrapper_from_dir(dirpath: str):
    with open(dirpath+'data.json') as jsonfile:
        data = json.load(jsonfile)
        wr = FamilyShapeSDFWrapper(path_to_training_npyfolder=data['path_to_training_npyfolder'],
                                   batch_size=data['batch_size'],
                                   h_blocks=data['h_blocks'],
                                   latent_size=data['latent_size'],
                                   path_to_saves=data['path_to_saves'])
        wr.model.load_state_dict(torch.load(dirpath+"model-parameters.pt"))
    return wr


def main():
    parser = get_parser()
    args = parser.parse_args()

    folderpath = args.input
    n_epochs = args.epochs
    latent_size = args.latent
    batch_size = args.batch
    h_blocks = args.height
    debug = args.debug
    learning_rate = args.rate
    workdir = f"e{n_epochs}_l{latent_size}_b{batch_size}_h{h_blocks}_lr{learning_rate}/"
    if path.exists(workdir):
        print(f"Workdir exists! {workdir}")
    else:
        mkdir(workdir)
        print(f"Created workdir:\t {workdir}")

    wrapper = FamilyShapeSDFWrapper(path_to_training_npyfolder=folderpath,
                                    batch_size=batch_size,
                                    h_blocks=h_blocks,
                                    latent_size=latent_size,
                                    path_to_saves=workdir)
    wrapper.train(n_epochs=n_epochs, learning_rate=learning_rate, debug=debug)
    wrapper.validate()
    wrapper.visualize_id_voxels(latent_id=wrapper.get_id_by_filename("0.npy"))
    wrapper.visualize_id_marchingcubes(latent_id=wrapper.get_id_by_filename("0.npy"))
    wrapper.plot_history()
    wrapper.save()


if __name__ == '__main__':
    main()
