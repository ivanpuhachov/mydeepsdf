import numpy as np
import torch

def get_torchgrid(grid_res=20, device='cuda'):
    grid = np.array(
        [[i // (grid_res ** 2), (i // grid_res) % grid_res, i % grid_res]
         for i in range(grid_res ** 3)], dtype=np.float)
    grid = ((grid - grid_res / 2) / grid_res) * 2.0
    grid = torch.from_numpy(np.float32(grid)).to(device)
    return grid

def get_sdfgrid(model, grid_res=20, device='cuda'):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid = get_torchgrid(grid_res, device)
    with torch.no_grad():
        outs = model.forward(grid).cpu().reshape(shape=(grid_res, grid_res, grid_res)).numpy()
    return outs