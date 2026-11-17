from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mesh import BoundaryFace
from .sem import SpectralGrid


@dataclass(frozen=True)
class FluidSolidInterface:
    fluid_nodes: NDArray[np.intp]
    solid_nodes: NDArray[np.intp]
    surface_weights: NDArray[np.float64]
    solid_normal: NDArray[np.float64]  # Outward from solid, into fluid.

    def add_fluid_load(
        self, load: NDArray[np.float64], displacement: NDArray[np.float64]
    ) -> None:
        normal_displacement = displacement[self.solid_nodes] @ self.solid_normal
        # The fluid's outward normal is opposite to the solid's.
        np.add.at(load[:, 0], self.fluid_nodes, -self.surface_weights * normal_displacement)

    def add_solid_load(
        self, load: NDArray[np.float64], pressure: NDArray[np.float64]
    ) -> None:
        traction = -pressure[self.fluid_nodes] * self.solid_normal
        np.add.at(load, self.solid_nodes, self.surface_weights[:, None] * traction)


def planar_interface(
    fluid: SpectralGrid,
    fluid_faces: tuple[BoundaryFace, ...],
    solid: SpectralGrid,
    solid_faces: tuple[BoundaryFace, ...],
) -> FluidSolidInterface:
    """Pair opposite, matching GLL faces from the two axis-aligned grids."""
    if not fluid_faces or len(fluid_faces) != len(solid_faces):
        raise ValueError("the interface requires the same positive number of faces on both sides")
    if len(set(fluid_faces)) != len(fluid_faces) or len(set(solid_faces)) != len(solid_faces):
        raise ValueError("interface faces must be unique on each side")
    axis, fluid_side = fluid_faces[0].axis, fluid_faces[0].side
    if axis not in (0, 1, 2) or fluid_side not in (0, 1):
        raise ValueError("invalid interface axis or side")
    solid_side = 1 - fluid_side
    tangents = [direction for direction in range(3) if direction != axis]

    def ordered_faces(
        grid: SpectralGrid, faces: tuple[BoundaryFace, ...], side: int
    ) -> list[BoundaryFace]:
        if any(
            face.axis != axis or face.side != side or not 0 <= face.element < len(grid.lower)
            for face in faces
        ):
            raise ValueError("interface faces must have valid elements and opposite orientations")
        return sorted(faces, key=lambda face: tuple(grid.lower[face.element, tangents]))

    ordered_fluid = ordered_faces(fluid, fluid_faces, fluid_side)
    ordered_solid = ordered_faces(solid, solid_faces, solid_side)
    if len(fluid.rule.nodes) != len(solid.rule.nodes):
        raise ValueError("interface faces must use the same GLL degree")

    def face_nodes(
        grid: SpectralGrid, faces: list[BoundaryFace], side: int
    ) -> NDArray[np.intp]:
        elements = [face.element for face in faces]
        return np.take(grid.local_to_global[elements], -1 if side else 0, axis=axis + 1)

    fluid_nodes = face_nodes(fluid, ordered_fluid, fluid_side)
    solid_nodes = face_nodes(solid, ordered_solid, solid_side)
    fluid_points = fluid.coordinates[fluid_nodes]
    solid_points = solid.coordinates[solid_nodes]
    tolerance = 32 * np.finfo(float).eps * max(1.0, np.abs(fluid_points).max())
    plane = fluid_points[..., axis]
    if (
        not np.allclose(plane, plane.flat[0], rtol=0, atol=tolerance)
        or not np.allclose(fluid_points, solid_points, rtol=0, atol=tolerance)
    ):
        raise ValueError("interface GLL coordinates must match on one plane")

    elements = [face.element for face in ordered_fluid]
    half_widths = 0.5 * (fluid.upper[elements] - fluid.lower[elements])
    face_jacobian = half_widths[:, tangents].prod(axis=1)
    surface_weights = face_jacobian[:, None, None] * np.outer(fluid.rule.weights, fluid.rule.weights)
    normal = np.zeros(3)
    normal[axis] = 2 * solid_side - 1
    return FluidSolidInterface(
        fluid_nodes.ravel(), solid_nodes.ravel(), surface_weights.ravel(), normal
    )
