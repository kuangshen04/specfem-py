from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .sem import SpectralDomain


def pressure_from_potential_acceleration(
    potential_acceleration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert the second time derivative of the scalar potential to pressure in Pa."""
    return -potential_acceleration


@dataclass(frozen=True)
class AcousticMaterial:
    density: float
    vp: float

    def __post_init__(self) -> None:
        values = np.array([self.density, self.vp])
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("density and vp must be finite and positive")

    @property
    def bulk_modulus(self) -> float:
        return self.density * self.vp**2


@dataclass(frozen=True)
class AcousticOperator:
    domain: SpectralDomain
    mass: NDArray[np.float64]
    inverse_density: NDArray[np.float64]

    def stiffness(self, potential: NDArray[np.float64]) -> NDArray[np.float64]:
        gradient = self.domain.gradient(self.domain.gather(potential))
        flux = self.inverse_density[:, None, None, None, None, None] * gradient
        # A rigid wall has zero normal flux and contributes no boundary integral.
        return self.domain.assemble(self.domain.integrate_gradient(flux))

    def acceleration(
        self, potential: NDArray[np.float64], load: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return (load - self.stiffness(potential)) / self.mass[:, None]


def acoustic_operator(
    domain: SpectralDomain,
    materials: Mapping[int, AcousticMaterial],
) -> AcousticOperator:
    region_ids = domain.grid.mesh.region_ids[domain.element_ids]
    missing = set(region_ids).difference(materials)
    if missing:
        raise ValueError(f"undefined material regions: {sorted(missing)}")
    element_materials = [materials[region] for region in region_ids]
    inverse_bulk = np.array([1 / material.bulk_modulus for material in element_materials])
    mass = domain.assemble(inverse_bulk[:, None, None, None] * domain.volume_weights)
    inverse_density = np.array([1 / material.density for material in element_materials])
    return AcousticOperator(domain, mass, inverse_density)
