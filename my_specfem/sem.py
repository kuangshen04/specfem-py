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
    rule: GLLRule
    coordinates: NDArray[np.float64]
    local_to_global: NDArray[np.intp]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    inverse_jacobian: NDArray[np.float64]
    volume_weights: NDArray[np.float64]

    @property
    def node_count(self) -> int:
        return len(self.coordinates)

    def gather(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        return field[self.local_to_global]

    def assemble(self, local_values: NDArray[np.float64]) -> NDArray[np.float64]:
        trailing = local_values.shape[self.local_to_global.ndim :]
        result = np.zeros((self.node_count, *trailing), dtype=local_values.dtype)
        np.add.at(result, self.local_to_global, local_values)
        return result

    def gradient(self, local_field: NDArray[np.float64]) -> NDArray[np.float64]:
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
        return reference_gradient * self.inverse_jacobian[:, None, None, None, None, :]

    def integrate_gradient(self, flux: NDArray[np.float64]) -> NDArray[np.float64]:
        """Integrate the physical test-function gradient against the given flux."""
        weighted_flux = (
            flux
            * self.volume_weights[..., None, None]
            * self.inverse_jacobian[:, None, None, None, None, :]
        )
        d = self.rule.derivative
        return (
            np.einsum("li,eljkc->eijkc", d, weighted_flux[..., 0])
            + np.einsum("lj,eilkc->eijkc", d, weighted_flux[..., 1])
            + np.einsum("lk,eijlc->eijkc", d, weighted_flux[..., 2])
        )

    def locate(self, position: tuple[float, float, float]) -> PointStencil:
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
        return PointStencil(self.local_to_global[element].ravel(), weights.ravel())


def discretize(mesh: HexMesh, degree: int = 4) -> SpectralGrid:
    """Build the GLL grid for a structured, axis-aligned mesh from box_mesh."""
    rule = gll_rule(degree)
    cells = np.indices(mesh.cell_shape).reshape(3, -1).T
    local = np.indices((degree + 1,) * 3).reshape(3, -1).T
    node_shape = tuple(n * degree + 1 for n in mesh.cell_shape)
    indices = cells[:, None, :] * degree + local
    flat_mapping = np.ravel_multi_index(tuple(indices.transpose(2, 0, 1)), node_shape)
    local_to_global = flat_mapping.reshape(-1, degree + 1, degree + 1, degree + 1)

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
        rule, coordinates, local_to_global, lower, upper, 1 / half_width, volume_weights
    )
