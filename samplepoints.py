from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import numpy as np
import argparse
from os import path, mkdir
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to parent folder of obj. Default: 'data/airplanes/'", default='data/airplanes')
parser.add_argument("-n", type=int, help="Number of points to sample. Default: 200000", default=200000)
args = parser.parse_args()

folderpath = args.input
n_points = args.n

if folderpath[:-1] != '/':
    print(folderpath)
    folderpath += '/'

objfolderpath = folderpath + 'obj/'
npyfolderpath = folderpath + 'npy/'

assert(path.exists(folderpath))
assert(path.exists(objfolderpath))
mkdir(npyfolderpath)

for i, objfilename in enumerate(os.listdir(objfolderpath)):
    try:
        # TODO: increadibly slow, needs optimizing
        filename = objfilename[:objfilename.find(".obj")]
        mesh = trimesh.load(objfolderpath+objfilename)
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=n_points)
        with open(npyfolderpath + filename + '.npy', 'wb') as f:
            np.save(f, points)
            np.save(f, sdf)
        print(f"{i + 1} files processed: {objfilename}")
    except:
        # TODO: investigate the source of problems with processing
        print(f" - Problems processing file {objfilename}")