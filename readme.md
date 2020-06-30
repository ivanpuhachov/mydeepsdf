# DeepSDF
Paper https://arxiv.org/abs/1901.05103

GitHub https://github.com/facebookresearch/DeepSDF

## TODO
 * investigate sampling technique
 * add instructions to processing meshes

## Point Sampling
Points are sampled by [mesh-to-sdf](https://github.com/marian42/mesh_to_sdf) package. 

## Data
Shapenet train: http://shapenet.cs.stanford.edu/shrec16/

### Rename
To rename all files in directory by their id:

```ls -v | cat -n | while read n f; do mv -n "$f" "$n.ext"; done ```