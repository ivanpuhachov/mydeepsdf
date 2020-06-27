from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np

mesh = trimesh.load('data/sphere.obj')

points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)

with open("data/sphere.npy", 'wb') as f:
    np.save(f, points)
    np.save(f, sdf)

colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)