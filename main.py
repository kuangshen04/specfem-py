from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from my_specfem.acoustic import AcousticMaterial, acoustic_operator
from my_specfem.acquisition import AcousticPointSource, PointForce, Receiver, Ricker, Seismogram
from my_specfem.coupling import fluid_solid_interface
from my_specfem.elastic import IsotropicMaterial, elastic_operator
from my_specfem.forward import run_acoustic, run_coupled, run_elastic
from my_specfem.io import write_seismogram
from my_specfem.mesh import box_mesh
from my_specfem.sem import SpectralDomain, discretize, spectral_domain


SOURCE_POSITION = (87.0, 86.0, 87.0)


def _box_setup() -> tuple[SpectralDomain, tuple[Receiver, ...]]:
    mesh = box_mesh((4, 2, 2), (400.0, 200.0, 200.0))
    domain = spectral_domain(discretize(mesh))
    receivers = (
        Receiver("STA01", domain.locate((143.0, 92.0, 88.0))),
        Receiver("STA02", domain.locate((269.0, 114.0, 93.0))),
    )
    return domain, receivers


def homogeneous_box(*, dt: float = 0.001, steps: int = 200) -> Seismogram:
    domain, receivers = _box_setup()
    materials = {0: IsotropicMaterial(density=2500.0, vp=3000.0, vs=1700.0)}
    operator = elastic_operator(domain, materials)
    source = PointForce(
        domain.locate(SOURCE_POSITION), np.array([1.0e8, 0.0, 0.0]), Ricker(20.0)
    )
    return run_elastic(operator, source, receivers, t0=-0.075, dt=dt, steps=steps)


def acoustic_box(*, dt: float = 0.001, steps: int = 200) -> Seismogram:
    domain, receivers = _box_setup()
    materials = {0: AcousticMaterial(density=1000.0, vp=1500.0)}
    operator = acoustic_operator(domain, materials)
    source = AcousticPointSource(domain.locate(SOURCE_POSITION), 0.01, Ricker(20.0))
    return run_acoustic(operator, source, receivers, t0=-0.075, dt=dt, steps=steps)


def coupled_box() -> tuple[Seismogram, Seismogram]:
    regions = np.zeros((48, 1, 1), dtype=np.intp)
    regions[24:] = 1
    mesh = box_mesh(regions.shape, (1200.0, 80.0, 80.0), regions)
    grid = discretize(mesh)
    water = AcousticMaterial(density=1000.0, vp=1500.0)
    # Zero lambda makes the sides traction-free for a planar longitudinal wave.
    solid_vp = 3000.0
    material = IsotropicMaterial(density=2500.0, vp=solid_vp, vs=solid_vp / np.sqrt(2))
    fluids, solids = {0: water}, {1: material}
    fluid = spectral_domain(grid, np.flatnonzero(np.isin(mesh.region_ids, tuple(fluids))))
    solid = spectral_domain(grid, np.flatnonzero(np.isin(mesh.region_ids, tuple(solids))))
    interface = fluid_solid_interface(fluid, solid)
    acoustic = acoustic_operator(fluid, fluids)
    elastic = elastic_operator(solid, solids)
    fluid_receivers = (Receiver("FLUID", fluid.locate((481.0, 29.0, 31.0))),)
    solid_receivers = (Receiver("SOLID", solid.locate((751.0, 29.0, 31.0))),)

    # A right-going plane pulse starts in the fluid, away from both x faces.
    width = 35.0
    offset = (fluid.coordinates[:, :1] - 300.0) / width
    potential = np.exp(-0.5 * offset**2)
    potential_rate = (water.vp / width) * offset * potential
    return run_coupled(
        acoustic, elastic, interface, fluid_receivers, solid_receivers,
        t0=0.0, dt=0.0003, steps=1400,
        initial_potential=potential, initial_potential_rate=potential_rate,
    )


def main() -> None:
    parser = ArgumentParser(description="Run a built-in 3D spectral-element example.")
    parser.add_argument("--case", choices=("elastic", "acoustic", "coupled"), default="elastic")
    args = parser.parse_args()
    if args.case == "acoustic":
        outputs = (("acoustic_seismogram.csv", acoustic_box()),)
    elif args.case == "coupled":
        pressure, displacement = coupled_box()
        outputs = (
            ("coupled_pressure.csv", pressure),
            ("coupled_displacement.csv", displacement),
        )
    else:
        outputs = (("seismogram.csv", homogeneous_box()),)
    directory = Path("output")
    directory.mkdir(exist_ok=True)
    for filename, seismogram in outputs:
        path = directory / filename
        write_seismogram(path, seismogram)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
