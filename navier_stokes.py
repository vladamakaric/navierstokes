import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from numpy.typing import NDArray


@dataclass
class FluidCell:
    # Index in the 1D array of fluid cells.
    index: int
    # Index in the 2D grid, which also contains obstacle cells.
    grid_index: Tuple[int, int]
    # Cells on the boundary have a unit normal pointing into the boundary.
    # TODO: If more boundary-specific fields are needed, create a subtype BoundaryCell.
    non_boundary_index: int
    boundary_normal: Optional[NDArray] = None


# @dataclass
# class Boundary:


# @dataclass
# class Simulation:
#     width: int
#     height: int
#     grid: NDArray
#     fluid_cells: List[FluidCell]
#     fluid_cell_index: NDArray
#     projection_matrix: NDArray

#     def __init__(self, grid: NDArray):
#         self.height = grid.shape[0]
#         self.width = grid.shape[1]
#         self.fluid_cells = fluid_cells(grid)
#         self.projection_matrix = self.build_projection_matrix()

def fluid_cell_index(fluid_cells, grid_shape):
    index = np.empty(shape=grid_shape, dtype=FluidCell)
    for fluid_cell in fluid_cells:
        index[fluid_cell.grid_index] = fluid_cell
    return index


# def projection_b(fluid_cells):
#     b = np.zeros(len(fluid_cells))
#     for fluid_cell in fluid_cells:
#         if fluid_cell.boundary_normal is None:
#             b[fluid_cell.index] = divW * step**2
#         else:
#             b[fluid_cell.index] = np.dot(fluid_cell.boundary_normal, w) * step**2
#     return b


def projection_A(fluid_cells, fluid_cell_index, width):
    # step = 1
    divW = 0
    non_boundary_cells = [cell for cell in fluid_cells if cell.boundary_normal is None]
    print(f'NON bndr len pA: {len(non_boundary_cells )}')
    w = np.array([1, 0])
    b = np.zeros(len(non_boundary_cells))
    A = np.zeros(shape=(len(non_boundary_cells), len(non_boundary_cells)))
    for fluid_cell in non_boundary_cells:
        # One linear equation per non-boundary fluid cell.
        row = A[fluid_cell.non_boundary_index]
        j, i = fluid_cell.grid_index
        row[fluid_cell.non_boundary_index] = -4
        b[fluid_cell.non_boundary_index] += divW
        # if j == 5 and i == 5:
        #     b[fluid_cell.non_boundary_index] += 1
        for j_offset, i_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cj = j + j_offset
            ci = (i + i_offset) % width
            cell = fluid_cell_index[cj][ci]
            if cell.boundary_normal is None:
                row[cell.non_boundary_index] = 1
            else:
                normal = cell.boundary_normal
                x_dir = -int(np.sign(normal[0]))
                y_dir = -int(np.sign(normal[1]))
                an_x = np.abs(normal[0])
                an_y = np.abs(normal[1])
                # This formula is correct:
                # (g-other_x)*abs(nx) + (g-other_y)*abs(ny) = R
                # g = (other_x*abs(nx) + other_y*abs(ny) + b)/(np.abs(nx + ny))
                print((cj,ci))
                if x_dir:
                    other_x = fluid_cell_index[cj, (ci+x_dir)%width].non_boundary_index
                    row[other_x] += an_x/(an_x + an_y)
                if y_dir:
                    other_y = fluid_cell_index[cj + y_dir, ci].non_boundary_index
                    row[other_y] += an_y/(an_x + an_y)
                b[fluid_cell.non_boundary_index] += -np.dot(w, normal)
                # TODO: When this laplace stensil dips into a boundary (ghost) point,
                # just express that point in terms of its constraint (deriv), and plug it
                # into the stencil. This way, we solve the poisson problem for the fluid
                # non-ghost cells! This is what ChatGPT told me, and I get it now.
    return A, b


def fluid_cells2(grid):
    height, width = grid.shape
    cells = []
    count = 0
    non_boundary_cell_count = 0
    for j, i in np.ndindex(grid.shape):
        boundary_normal = None
        if j == 0:
            boundary_normal = np.array([0, -1])
        elif j == height - 1:
            boundary_normal = np.array([0, 1])
        elif i > 0 and i < width - 1:
            if grid[j][i]:
                empty_neighbors = [
                    (jd, id)
                    for (jd, id) in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    if not grid[j + jd][(i + id) % width]
                ]
                assert len(empty_neighbors) < 3
                if len(empty_neighbors) == 0:
                    continue
                boundary_normal = -np.sum(
                    [np.flip(np.array(en)) for en in empty_neighbors], axis=0
                )
                print(boundary_normal)
                assert boundary_normal.any()  # [0,0] means one point thick boundary
                boundary_normal = boundary_normal / np.linalg.norm(boundary_normal)
        cells += [
            FluidCell(
                index=count,
                grid_index=(j, i),
                non_boundary_index = non_boundary_cell_count,
                boundary_normal=boundary_normal
            )
        ]
        if boundary_normal is None:
            non_boundary_cell_count +=1
        count += 1
    print(f'Non bnd count: {non_boundary_cell_count}')
    return cells


def fluid_cells(grid, inward_corner_difference, simple_normals):
    count = 0
    cells = []
    height, width = grid.shape
    index = np.ones(shape=grid.shape, dtype=int) * -1
    count = 0
    for j, i in np.ndindex(grid.shape):
        if grid[j][i]:
            continue
        index[j][i] = count
        count += 1

    for j, i in np.ndindex(grid.shape):
        if grid[j][i]:
            continue
        x_diff = (1, -1)
        y_diff = (1, -1)
        boundary_normal = None
        if j == 0:
            y_diff = (1, 0)
            boundary_normal = np.array([0, -1])
        elif j == height - 1:
            y_diff = (0, -1)
            boundary_normal = np.array([0, 1])
        elif i > 0 and i < width - 1:
            neighborhood = grid[j - 1 : j + 2, i - 1 : i + 2]
            boundary_normal, x_diff, y_diff = normal_and_diffs(
                neighborhood, inward_corner_difference, simple_normals
            )
        cells += [
            FluidCell(
                index=index[j][i],
                grid_index=(j, i),
                boundary_normal=boundary_normal,
                x_diff=(
                    index[j][(i + x_diff[0]) % width],
                    index[j][(i + x_diff[1]) % width],
                    x_diff[0] - x_diff[1],
                ),
                y_diff=(
                    index[j + y_diff[0]][i],
                    index[j + y_diff[1]][i],
                    y_diff[0] - y_diff[1],
                ),
            )
        ]
    return cells


def normal_and_diffs(neighborhood, inward_corner_difference, simple_normals):
    obstacle_indices = [
        (j - 1, i - 1) for j in range(3) for i in range(3) if neighborhood[j][i]
    ]
    # TODO: Add option for no outward corner.
    if len(obstacle_indices) == 0:
        return None, (1, -1), (1, -1)

    if len(obstacle_indices) == 1 and obstacle_indices[0][0] and obstacle_indices[0][1]:
        j, i = obstacle_indices[0]
        boundary_normal = np.array([i, j]) / np.sqrt(2)
        if inward_corner_difference:
            x_diff = (max(0, i), min(0, i))
            y_diff = (max(0, j), min(0, j))
        else:
            x_diff = (max(0, -i), min(0, -i))
            y_diff = (max(0, -j), min(0, -j))
        return boundary_normal, x_diff, y_diff

    empty_i = [i for i in [0, 1, 2] if not neighborhood[1][i]]
    empty_j = [j for j in [0, 1, 2] if not neighborhood[j][1]]
    x_diff = (max(empty_i) - 1, min(empty_i) - 1)
    y_diff = (max(empty_j) - 1, min(empty_j) - 1)

    if simple_normals:
        # Concave corner
        if (x_diff[0] - x_diff[1]) == 1 and (y_diff[0] - y_diff[1]) == 1:
            # Normal is opposite the difference
            free_x_dir = x_diff[0] + x_diff[1]
            free_y_dir = y_diff[0] + y_diff[1]
            boundary_normal = -np.array([free_x_dir, free_y_dir]) / np.sqrt(2)
        else:
            if (x_diff[0] - x_diff[1]) == 1:
                free_x_dir = x_diff[0] + x_diff[1]
                boundary_normal = np.array([-free_x_dir, 0])
            else:
                free_y_dir = y_diff[0] + y_diff[1]
                boundary_normal = np.array([0, -free_y_dir])
    else:
        boundary_normal = local_boundary_normal(neighborhood)

    return boundary_normal, x_diff, y_diff


# def boundary_normal(j, i, grid):
#     if j == 0:
#         return np.array([0, -1])
#     if j == height - 1:
#         return np.array([0, 1])
#     if i == 0 or i == width - 1:
#         return None
#     if not np.any(neighborhood):
#         return None
#     return local_boundary_normal(neighborhood)


def local_boundary_normal(neighborhood3x3):
    normal = np.array([0, 0])
    for j in range(-1, 2):
        for i in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if neighborhood3x3[j + 1, i + 1]:
                normal += np.array([i, j])
    return normal / np.linalg.norm(normal)


def simplify_normals(fluid_cells):
    for fluid_cell in fluid_cells:
        if fluid_cell.boundary_normal is None:
            continue
        n_x, n_y = fluid_cell.boundary_normal[0], fluid_cell.boundary_normal[1]
        if n_x == 0 or n_y == 0:
            continue
        if np.abs(n_x) == np.abs(n_y):
            continue
        if np.abs(n_x) > np.abs(n_y):
            fluid_cell.boundary_normal = np.array([np.sign(n_x), 0])
        else:
            fluid_cell.boundary_normal = np.array([0, np.sign(n_y)])
