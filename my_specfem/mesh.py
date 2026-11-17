from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray


# Corner order follows the tensor-product axes x, y, z, with z fastest.
_CORNERS = np.array(list(product(range(2), repeat=3)))


@dataclass(frozen=True)
class BoundaryFace:
    element: int
    axis: int
    side: int  # 0 is the lower face, 1 is the upper face.


@dataclass(frozen=True)
class HexMesh:
    vertices: NDArray[np.float64]
    connectivity: NDArray[np.intp]
    cell_shape: tuple[int, int, int]
    region_ids: NDArray[np.intp]
    boundary_faces: tuple[BoundaryFace, ...]


def box_mesh(
    cell_shape: tuple[int, int, int],
    size: tuple[float, float, float],
    region_ids: NDArray[np.intp] | None = None,
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> HexMesh:
    """Generate a conforming axis-aligned mesh with the given lower corner."""
    if len(cell_shape) != 3 or any(
        not isinstance(n, (int, np.integer)) or n < 1 for n in cell_shape
    ):
        raise ValueError("cell_shape must contain three positive integers")
    lengths = np.asarray(size, dtype=float)
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError("size must contain three finite positive lengths")
    lower = np.asarray(origin, dtype=float)
    if lower.shape != (3,) or not np.isfinite(lower).all():
        raise ValueError("origin must contain three finite coordinates")

    vertex_shape = tuple(n + 1 for n in cell_shape)
    axes = [
        start + np.linspace(0.0, length, n)
        for start, length, n in zip(lower, lengths, vertex_shape)
    ]
    if any(not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0) for axis in axes):
        raise ValueError("box vertices must be finite and strictly increasing on each axis")
    vertices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    cells = np.indices(cell_shape).reshape(3, -1).T
    corners = cells[:, None, :] + _CORNERS
    connectivity = np.ravel_multi_index(tuple(corners.transpose(2, 0, 1)), vertex_shape)
    if region_ids is None:
        regions = np.zeros(len(cells), dtype=np.intp)
    else:
        regions = np.asarray(region_ids)
        if regions.shape != cell_shape or regions.dtype.kind not in "iu":
            raise ValueError("region_ids must be an integer array shaped like cell_shape")
        regions = regions.astype(np.intp, copy=True).reshape(-1)

    faces = tuple(
        BoundaryFace(element, axis, side)
        for element, cell in enumerate(cells)
        for axis, n in enumerate(cell_shape)
        for side, index in enumerate((0, n - 1))
        if cell[axis] == index
    )
    return HexMesh(vertices, connectivity, cell_shape, regions, faces)
