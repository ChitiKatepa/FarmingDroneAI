#how the process of putting a heatmap
#on the grid should look

import numpy as np

# mask: 2D array from segmentation
# 0 = healthy, 1 = mild, 2 = severe

severity_weights = {
    0: 0.0,
    1: 0.5,
    2: 1,
}

severity_map = np.vectorize(severity_weights.get)(mask)

grid_size = 20  # pixels per grid cell
h, w = severity_map.shape

grid = []

for i in range(0, h, grid_size):
    row = []
    for j in range(0, w, grid_size):
        cell = severity_map[i:i+grid_size, j:j+grid_size]
        row.append(np.mean(cell))
    grid.append(row)

grid = np.array(grid)
