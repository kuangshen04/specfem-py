from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .sem import SpectralDomain


@dataclass(frozen=True)
class FluidSolidInterface:
    fluid_nodes: NDArray[np.intp]
    solid_nodes: NDArray[np.intp]
    surface_weights: NDArray[np.float64]
    solid_normals: NDArray[np.float64]  # One per face quadrature point, outward from solid.

    def add_fluid_load(
        self, load: NDArray[np.float64], displacement: NDArray[np.float64]
    ) -> None:
        normal_displacement = np.einsum(
            "ni,ni->n", displacement[self.solid_nodes], self.solid_normals
        )
        # The fluid's outward normal is opposite to the solid's.
        np.add.at(load[:, 0], self.fluid_nodes, -self.surface_weights * normal_displacement)

    def add_solid_load(
        self, load: NDArray[np.float64], pressure: NDArray[np.float64]
    ) -> None:
        traction = -pressure[self.fluid_nodes] * self.solid_normals
        np.add.at(load, self.solid_nodes, self.surface_weights[:, None] * traction)


def fluid_solid_interface(
    fluid: SpectralDomain, solid: SpectralDomain
) -> FluidSolidInterface:
    """Build field-node pairs from interior faces that separate fluid and solid."""
    grid = fluid.grid
    if solid.grid is not grid:
        raise ValueError("fluid and solid domains must share a spectral grid")
    in_fluid = fluid.row_for_element >= 0
    in_solid = solid.row_for_element >= 0
    if np.any(in_fluid == in_solid):
        raise ValueError("each mesh element must belong to exactly one coupled domain")

    fluid_nodes = []
    solid_nodes = []
    weights = []
    normals = []
    for face in grid.mesh.interior_faces:
        lower, upper, axis = face.lower_element, face.upper_element, face.axis
        if in_fluid[lower] == in_fluid[upper]:
            continue
        if in_fluid[lower]:
            fluid_element, fluid_face = lower, -1
            solid_element, solid_face, sign = upper, 0, -1
        else:
            fluid_element, fluid_face = upper, 0
            solid_element, solid_face, sign = lower, -1, 1

        fluid_row = fluid.row_for_element[fluid_element]
        solid_row = solid.row_for_element[solid_element]
        fluid_nodes.append(np.take(fluid.element_to_field_node[fluid_row], fluid_face, axis).ravel())
        solid_nodes.append(np.take(solid.element_to_field_node[solid_row], solid_face, axis).ravel())
        face_weights = grid.face_weights(lower, axis).ravel()
        normal = np.zeros((len(face_weights), 3))
        normal[:, axis] = sign
        weights.append(face_weights)
        normals.append(normal)

    if not weights:
        raise ValueError("fluid and solid domains must meet on an interior face")
    return FluidSolidInterface(
        np.concatenate(fluid_nodes), np.concatenate(solid_nodes),
        np.concatenate(weights), np.concatenate(normals),
    )
