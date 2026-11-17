from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .sem import SpectralGrid


@dataclass(frozen=True)
class IsotropicMaterial:
    density: float
    vp: float
    vs: float

    def __post_init__(self) -> None:
        values = np.array([self.density, self.vp, self.vs])
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("density, vp and vs must be finite and positive")
        if self.bulk_modulus <= 0:
            raise ValueError("bulk modulus must be positive")

    @property
    def shear_modulus(self) -> float:
        return self.density * self.vs**2

    @property
    def bulk_modulus(self) -> float:
        return self.density * self.vp**2 - (4 / 3) * self.shear_modulus

    @property
    def lame_lambda(self) -> float:
        return self.bulk_modulus - (2 / 3) * self.shear_modulus


def isotropic_stress(
    gradient: NDArray[np.float64],
    lame_lambda: NDArray[np.float64],
    shear_modulus: NDArray[np.float64],
) -> NDArray[np.float64]:
    volumetric_strain = np.trace(gradient, axis1=-2, axis2=-1)
    symmetric_gradient = gradient + gradient.swapaxes(-1, -2)
    return (
        (lame_lambda * volumetric_strain)[..., None, None] * np.eye(3)
        + shear_modulus[..., None, None] * symmetric_gradient
    )


@dataclass(frozen=True)
class ElasticOperator:
    grid: SpectralGrid
    mass: NDArray[np.float64]
    lame_lambda: NDArray[np.float64]
    shear_modulus: NDArray[np.float64]

    def stiffness(self, displacement: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Ku by integrating stress against the test-function gradient."""
        gradient = self.grid.gradient(self.grid.gather(displacement))
        stress = isotropic_stress(
            gradient,
            self.lame_lambda[:, None, None, None],
            self.shear_modulus[:, None, None, None],
        )
        # The boundary integral is zero on every traction-free exterior face.
        return self.grid.assemble(self.grid.integrate_gradient(stress))

    def acceleration(
        self, displacement: NDArray[np.float64], force: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return (force - self.stiffness(displacement)) / self.mass[:, None]


def elastic_operator(
    grid: SpectralGrid,
    region_ids: NDArray[np.intp],
    materials: Mapping[int, IsotropicMaterial],
) -> ElasticOperator:
    if region_ids.shape != (len(grid.lower),):
        raise ValueError("one material region is required per element")
    missing = set(region_ids).difference(materials)
    if missing:
        raise ValueError(f"undefined material regions: {sorted(missing)}")
    element_materials = [materials[region] for region in region_ids]
    density = np.array([material.density for material in element_materials])
    mass = grid.assemble(density[:, None, None, None] * grid.volume_weights)
    lam = np.array([material.lame_lambda for material in element_materials])
    mu = np.array([material.shear_modulus for material in element_materials])
    return ElasticOperator(grid, mass, lam, mu)
