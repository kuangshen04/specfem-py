from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import Legendre
from numpy.typing import NDArray

from .mesh import HexMesh


@dataclass(frozen=True)
class GLLRule:
    nodes: NDArray[np.float64]
    weights: NDArray[np.float64]
    derivative: NDArray[np.float64]
    barycentric_weights: NDArray[np.float64]

    def basis(self, reference: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the 1D basis at each reference coordinate."""
        offsets = reference[:, None] - self.nodes
        values = np.empty_like(offsets)
        for axis, row in enumerate(offsets):
            hits = np.abs(row) <= 8 * np.finfo(float).eps
            if hits.any():
                values[axis] = hits
            else:
                terms = self.barycentric_weights / row
                values[axis] = terms / terms.sum()
        return values


def gll_rule(degree: int) -> GLLRule:
    if not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("GLL degree must be a positive integer")
    polynomial = Legendre.basis(degree)
    interior = np.real_if_close(polynomial.deriv().roots())
    if np.iscomplexobj(interior):
        raise ValueError("GLL nodes must be real")
    nodes = np.r_[-1.0, interior, 1.0]
    weights = 2.0 / (degree * (degree + 1) * polynomial(nodes) ** 2)
    offsets = nodes[:, None] - nodes
    np.fill_diagonal(offsets, 1.0)
    barycentric = 1.0 / offsets.prod(axis=1)
    derivative = barycentric[None, :] / (barycentric[:, None] * offsets)
    np.fill_diagonal(derivative, 0.0)
    np.fill_diagonal(derivative, -derivative.sum(axis=1))
    return GLLRule(nodes, weights, derivative, barycentric)


@dataclass(frozen=True)
class PointStencil:
    nodes: NDArray[np.intp]
    weights: NDArray[np.float64]

    def sample(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.weights @ field[self.nodes]

    def add_load(self, load: NDArray[np.float64], value: NDArray[np.float64]) -> None:
        np.add.at(load, self.nodes, self.weights[:, None] * value)


@dataclass(frozen=True)
class SpectralGrid:
    """SEM geometry for the whole mesh."""

    mesh: HexMesh
    rule: GLLRule
    coordinates: NDArray[np.float64]
    element_to_geometry_node: NDArray[np.intp]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    inverse_jacobian: NDArray[np.float64]
    volume_weights: NDArray[np.float64]

    @property
    def node_count(self) -> int:
        return len(self.coordinates)

    def gradient(
        self, local_field: NDArray[np.float64], elements: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """Map (element, i, j, k, component) to physical gradients with an extra axis."""
        d = self.rule.derivative
        reference_gradient = np.stack(
            [
                np.einsum("il,eljkc->eijkc", d, local_field),
                np.einsum("jl,eilkc->eijkc", d, local_field),
                np.einsum("kl,eijlc->eijkc", d, local_field),
            ],
            axis=-1,
        )
        return reference_gradient * self.inverse_jacobian[elements, None, None, None, None, :]

    def integrate_gradient(
        self, flux: NDArray[np.float64], elements: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """Integrate the physical test-function gradient against the given flux."""
        weighted_flux = (
            flux
            * self.volume_weights[elements, ..., None, None]
            * self.inverse_jacobian[elements, None, None, None, None, :]
        )
        d = self.rule.derivative
        return (
            np.einsum("li,eljkc->eijkc", d, weighted_flux[..., 0])
            + np.einsum("lj,eilkc->eijkc", d, weighted_flux[..., 1])
            + np.einsum("lk,eijlc->eijkc", d, weighted_flux[..., 2])
        )

    def face_weights(self, element: int, axis: int) -> NDArray[np.float64]:
        tangents = [direction for direction in range(3) if direction != axis]
        half_widths = 0.5 * (self.upper[element] - self.lower[element])
        return half_widths[tangents].prod() * np.outer(self.rule.weights, self.rule.weights)

    def locate(self, position: tuple[float, float, float]) -> tuple[int, NDArray[np.float64]]:
        """Return the mesh element ID and its local interpolation weights."""
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("position must contain three finite coordinates")
        matches = np.flatnonzero(
            np.all((self.lower < point) & (point < self.upper), axis=1)
        )
        if len(matches) != 1:
            raise ValueError(f"position {position} must lie strictly inside one element")
        element = matches[0]
        reference = (point - self.lower[element]) * self.inverse_jacobian[element] - 1
        x, y, z = self.rule.basis(reference)
        weights = np.einsum("i,j,k->ijk", x, y, z)
        return int(element), weights.ravel()


@dataclass(frozen=True)
class SpectralDomain:
    """A subset of mesh elements with its own compact field node numbering."""

    grid: SpectralGrid
    element_ids: NDArray[np.intp]
    geometry_nodes: NDArray[np.intp]
    element_to_field_node: NDArray[np.intp]
    row_for_element: NDArray[np.intp]  # -1 for an element outside the domain.

    @property
    def node_count(self) -> int:
        return len(self.geometry_nodes)

    @property
    def coordinates(self) -> NDArray[np.float64]:
        return self.grid.coordinates[self.geometry_nodes]

    @property
    def volume_weights(self) -> NDArray[np.float64]:
        return self.grid.volume_weights[self.element_ids]

    def gather(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        if field.shape[:1] != (self.node_count,):
            raise ValueError("the first field axis must match the domain node count")
        return field[self.element_to_field_node]

    def assemble(self, local_values: NDArray[np.float64]) -> NDArray[np.float64]:
        local_shape = self.element_to_field_node.shape
        if local_values.shape[:len(local_shape)] != local_shape:
            raise ValueError("local values must match the domain elements and GLL axes")
        trailing = local_values.shape[len(local_shape):]
        result = np.zeros((self.node_count, *trailing), dtype=local_values.dtype)
        np.add.at(result, self.element_to_field_node, local_values)
        return result

    def gradient(self, local_field: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.grid.gradient(local_field, self.element_ids)

    def integrate_gradient(self, flux: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.grid.integrate_gradient(flux, self.element_ids)

    def locate(self, position: tuple[float, float, float]) -> PointStencil:
        element, weights = self.grid.locate(position)
        row = self.row_for_element[element]
        if row < 0:
            raise ValueError(f"position {position} is outside this field domain")
        return PointStencil(self.element_to_field_node[row].ravel(), weights)


def spectral_domain(
    grid: SpectralGrid, element_ids: NDArray[np.intp] | None = None
) -> SpectralDomain:
    element_count = len(grid.element_to_geometry_node)
    elements = np.arange(element_count) if element_ids is None else np.asarray(element_ids)
    if (
        elements.ndim != 1 or elements.size == 0 or elements.dtype.kind not in "iu"
        or np.any(elements < 0) or np.any(elements >= element_count)
    ):
        raise ValueError("a field domain requires a nonempty list of valid mesh element IDs")
    elements = np.sort(elements.astype(np.intp, copy=False))
    if np.any(np.diff(elements) == 0):
        raise ValueError("a field domain cannot list an element twice")
    geometry = grid.element_to_geometry_node[elements]
    geometry_nodes, local_to_field = np.unique(geometry, return_inverse=True)
    rows = np.full(element_count, -1, dtype=np.intp)
    rows[elements] = np.arange(len(elements))
    return SpectralDomain(
        grid, elements, geometry_nodes, local_to_field.reshape(geometry.shape), rows
    )


def discretize(mesh: HexMesh, degree: int = 4) -> SpectralGrid:
    """Build the GLL grid for a structured, axis-aligned mesh from box_mesh."""
    rule = gll_rule(degree)
    cells = np.indices(mesh.cell_shape).reshape(3, -1).T
    local = np.indices((degree + 1,) * 3).reshape(3, -1).T
    node_shape = tuple(n * degree + 1 for n in mesh.cell_shape)
    indices = cells[:, None, :] * degree + local
    flat_mapping = np.ravel_multi_index(tuple(indices.transpose(2, 0, 1)), node_shape)
    element_to_geometry_node = flat_mapping.reshape(-1, degree + 1, degree + 1, degree + 1)

    lower = mesh.vertices[mesh.connectivity[:, 0]]
    upper = mesh.vertices[mesh.connectivity[:, -1]]
    half_width = 0.5 * (upper - lower)
    offsets = 1 + rule.nodes[local]
    element_coordinates = lower[:, None, :] + half_width[:, None, :] * offsets
    coordinates = np.empty((np.prod(node_shape), 3))
    coordinates[flat_mapping] = element_coordinates

    weights = np.einsum("i,j,k->ijk", rule.weights, rule.weights, rule.weights)
    volume_weights = half_width.prod(axis=1)[:, None, None, None] * weights
    return SpectralGrid(
        mesh, rule, coordinates, element_to_geometry_node,
        lower, upper, 1 / half_width, volume_weights,
    )
