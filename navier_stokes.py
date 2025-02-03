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
    x_diff: Tuple[int, int]
    y_diff: Tuple[int, int]
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
    index = np.ones(shape=grid_shape, dtype=int) * -1
    for fluid_cell in fluid_cells:
        index[fluid_cell.grid_index] = fluid_cell.index
    return index


def projection_b(fluid_cells):
    divW = 0
    step = 1
    w = np.array([2, 0])
    b = np.zeros(len(fluid_cells))
    for fluid_cell in fluid_cells:
        if fluid_cell.boundary_normal is None:
            b[fluid_cell.index] = divW * step**2
        else:
            b[fluid_cell.index] = np.dot(fluid_cell.boundary_normal, w) * step**2
    return b


def projection_A(fluid_cells, fluid_cell_index, width, height):
    A = np.zeros(shape=(len(fluid_cells), len(fluid_cells)))
    for fluid_cell in fluid_cells:
        # One linear equation per fluid cell.
        row = A[fluid_cell.index]
        j, i = fluid_cell.grid_index
        if fluid_cell.boundary_normal is None:
            row[fluid_cell.index] = -4
            for j_offset, i_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                row[fluid_cell_index[j + j_offset][(i + i_offset) % width]] = 1
        else:
            normal = fluid_cell.boundary_normal
            n_x = normal[0]
            n_y = normal[1]
            if n_x != 0:
                if n_x > 0:
                    # backward difference
                    row[fluid_cell.index] += n_x
                    row[fluid_cell_index[j][i - 1]] += -n_x
                else:
                    # forward difference
                    row[fluid_cell_index[j][i + 1]] += n_x
                    row[fluid_cell.index] += -n_x
                # xd = x_diff(fluid_cell, fluid_cell_index, width)
                # print(f'{fluid_cell.grid_index} xd{xd}')
                # row[xd[0]] += n_x/xd[2]
                # row[xd[1]] += -n_x/xd[2]
            if n_y != 0:
                if n_y > 0:
                    # backward difference
                    row[fluid_cell.index] += n_y
                    row[fluid_cell_index[j - 1][i]] -= n_y
                else:
                    # forward difference
                    row[fluid_cell_index[j + 1][i]] += n_y
                    row[fluid_cell.index] -= n_y
                # yd = y_diff(fluid_cell, fluid_cell_index, height)
                # print(f'{fluid_cell.grid_index} yd{yd}')
                # row[yd[0]] += n_y/yd[2]
                # row[yd[1]] += -n_y/yd[2]
    return A


# def x_diff2(fluid_cell, fluid_cell_index, width):
#     j,i = fluid_cell.grid_index
#     left_index = fluid_cell_index[j][(i-1)%width]
#     right_index = fluid_cell_index[j][(i+1)%width]
#     size = 2 
#     if right_index == -1:
#         right_index = fluid_cell.index
#         size = 1
#     elif left_index == -1:
#         left_index = fluid_cell.index
#         size = 1
#     return (right_index, left_index, size)


# def y_diff2(fluid_cell, fluid_cell_index, height):
#     j,i = fluid_cell.grid_index
#     down_index = fluid_cell.index
#     up_index = fluid_cell.index
#     size = 0
#     if j > 0 and fluid_cell_index[j-1][i] != -1:
#         down_index = fluid_cell_index[j-1][i] 
#         size += 1
#     if j < height-1 and fluid_cell_index[j+1][i] != -1:
#         up_index = fluid_cell_index[j+1][i] 
#         size += 1
#     return (up_index, down_index, size)


# def x_diff(fluid_cell, fluid_cell_index, width):
#     j,i = fluid_cell.grid_index
#     left_index = fluid_cell_index[j][(i-1)%width]
#     right_index = fluid_cell_index[j][(i+1)%width]
#     size = 2 
#     if right_index == -1:
#         right_index = fluid_cell.index
#         size = 1
#     elif left_index == -1:
#         left_index = fluid_cell.index
#         size = 1
#     return (right_index, left_index, size)


# def y_diff(fluid_cell, fluid_cell_index, height):
#     j,i = fluid_cell.grid_index
#     down_index = fluid_cell.index
#     up_index = fluid_cell.index
#     size = 0
#     if j > 0 and fluid_cell_index[j-1][i] != -1:
#         down_index = fluid_cell_index[j-1][i] 
#         size += 1
#     if j < height-1 and fluid_cell_index[j+1][i] != -1:
#         up_index = fluid_cell_index[j+1][i] 
#         size += 1
#     return (up_index, down_index, size)


    #     n_x = fluid_cell.boundary_normal[0]
    #     if n_x != 0:
    #         if n_x > 0:
    #             return (fluid_cell.index, fluid_cell_index[j][i-1],1)
    #         else:
    #             return (fluid_cell_index[j][i+1], fluid_cell.index, 1)
    #     else:
    #             return (fluid_cell_index[j][i+1], fluid_cell_index[j][i-1], 2)
    #     # TODO: the last else only works for non-siplified normals.
    # return (fluid_cell_index[j][i+1], fluid_cell_index[j][i-1], 2)




def fluid_cells(grid, inward_corner_difference, simple_normals):
    count = 0
    cells = []
    height, width = grid.shape
    index = np.ones(shape=grid.shape, dtype=int)*-1
    count = 0
    for j, i in np.ndindex(grid.shape):
        if grid[j][i]:
            continue
        index[j][i] = count
        count+=1

    for j, i in np.ndindex(grid.shape):
        if grid[j][i]:
            continue
        x_diff = (1, -1)
        y_diff = (1, -1)
        boundary_normal=None
        if j == 0:
            y_diff = (1,0)
            boundary_normal = np.array([0, -1])
        elif j == height - 1:
            y_diff = (0,-1)
            boundary_normal = np.array([0, 1])
        elif i > 0 and i < width-1:
            neighborhood = grid[j - 1 : j + 2, i - 1 : i + 2]
            boundary_normal, x_diff, y_diff = normal_and_diffs(
                neighborhood, inward_corner_difference, simple_normals)
        cells += [
            FluidCell(
                index=index[j][i],
                grid_index=(j, i),
                boundary_normal=boundary_normal,
                x_diff=(
                    index[j][(i+x_diff[0])%width],
                    index[j][(i+x_diff[1])%width],
                    x_diff[0] - x_diff[1]
                ),
                y_diff=(
                    index[j+y_diff[0]][i],
                    index[j+y_diff[1]][i],
                    y_diff[0] - y_diff[1]
                ),
            )
        ]
    return cells


def normal_and_diffs(neighborhood, inward_corner_difference, simple_normals):
    obstacle_indices = [(j-1,i-1) for j in range(3) for i in range(3) if neighborhood[j][i]]
    # TODO: Add option for no outward corner.
    if len(obstacle_indices) == 0:
        return None, (1, -1), (1, -1)

    if len(obstacle_indices) == 1 and obstacle_indices[0][0] and obstacle_indices[0][1]:
        j,i = obstacle_indices[0]
        boundary_normal = np.array([i,j])/np.sqrt(2)
        if inward_corner_difference:
            x_diff = (max(0,i), min(0,i))
            y_diff = (max(0,j), min(0,j))
        else:
            x_diff = (max(0,-i), min(0,-i))
            y_diff = (max(0,-j), min(0,-j))
        return boundary_normal, x_diff, y_diff

    empty_i = [i for i in [0,1,2] if not neighborhood[1][i]]
    empty_j = [j for j in [0,1,2] if not neighborhood[j][1]]
    x_diff = (max(empty_i)-1, min(empty_i)-1)
    y_diff = (max(empty_j)-1, min(empty_j)-1)

    if simple_normals:
        # Concave corner
        if (x_diff[0] - x_diff[1]) == 1 and (y_diff[0] - y_diff[1]) == 1:
            # Normal is opposite the difference
            free_x_dir = x_diff[0] + x_diff[1]
            free_y_dir = y_diff[0] + y_diff[1]
            boundary_normal = -np.array([free_x_dir, free_y_dir])/np.sqrt(2)
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
        n_x,n_y = fluid_cell.boundary_normal[0], fluid_cell.boundary_normal[1]  
        if n_x == 0 or n_y == 0:
            continue
        if np.abs(n_x) == np.abs(n_y):
            continue
        if np.abs(n_x) > np.abs(n_y):
            fluid_cell.boundary_normal = np.array([np.sign(n_x), 0])
        else:
            fluid_cell.boundary_normal = np.array([0, np.sign(n_y)])