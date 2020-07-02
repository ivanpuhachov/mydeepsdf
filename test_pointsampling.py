from mesh_to_sdf import sample_sdf_near_surface, scale_to_unit_sphere

import trimesh
import pyrender
import numpy as np
import meshplot
meshplot.offline()

mesh = trimesh.load('data/chair.obj')

points, sdf = sample_sdf_near_surface(mesh, number_of_points=300000)

with open("data/chair.npy", 'wb') as f:
    np.save(f, points)
    np.save(f, sdf)

# colors = np.zeros(points.shape)
# colors[sdf < 0, 2] = 1
# colors[sdf > 0, 0] = 1
# cloud = pyrender.Mesh.from_points(points, colors=colors)
# scene = pyrender.Scene()
# scene.add(cloud)
# viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

# Rendering with meshplot

mesh = scale_to_unit_sphere(mesh)
points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
p = meshplot.plot(mesh.vertices, mesh.faces, filename="debug/test_pointsampling.html")
p.add_points(points, c=sdf,
                shading={"point_size": 0.08,
                         "alpha": 0.3,
                         "normalize": [True, True],
                         "colormap": "jet"})
p.save("debug/test_pointsampling.html")