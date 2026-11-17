from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from my_specfem.acoustic import AcousticMaterial, acoustic_operator
from my_specfem.acquisition import AcousticPointSource, PointForce, Receiver, Ricker, Seismogram
from my_specfem.coupling import planar_interface
from my_specfem.elastic import IsotropicMaterial, elastic_operator
from my_specfem.forward import run_acoustic, run_coupled, run_elastic
from my_specfem.io import write_seismogram
from my_specfem.mesh import HexMesh, box_mesh
from my_specfem.sem import SpectralGrid, discretize


SOURCE_POSITION = (87.0, 86.0, 87.0)


def _box_setup() -> tuple[HexMesh, SpectralGrid, tuple[Receiver, ...]]:
    mesh = box_mesh((4, 2, 2), (400.0, 200.0, 200.0))
    grid = discretize(mesh)
    receivers = (
        Receiver("STA01", grid.locate((143.0, 92.0, 88.0))),
        Receiver("STA02", grid.locate((269.0, 114.0, 93.0))),
    )
    return mesh, grid, receivers


def homogeneous_box(*, dt: float = 0.001, steps: int = 200) -> Seismogram:
    mesh, grid, receivers = _box_setup()
    materials = {0: IsotropicMaterial(density=2500.0, vp=3000.0, vs=1700.0)}
    operator = elastic_operator(grid, mesh.region_ids, materials)
    source = PointForce(
        grid.locate(SOURCE_POSITION), np.array([1.0e8, 0.0, 0.0]), Ricker(20.0)
    )
    return run_elastic(operator, source, receivers, t0=-0.075, dt=dt, steps=steps)


def acoustic_box(*, dt: float = 0.001, steps: int = 200) -> Seismogram:
    mesh, grid, receivers = _box_setup()
    materials = {0: AcousticMaterial(density=1000.0, vp=1500.0)}
    operator = acoustic_operator(grid, mesh.region_ids, materials)
    source = AcousticPointSource(grid.locate(SOURCE_POSITION), 0.01, Ricker(20.0))
    return run_acoustic(operator, source, receivers, t0=-0.075, dt=dt, steps=steps)


def coupled_box() -> tuple[Seismogram, Seismogram]:
    shape = (24, 1, 1)
    size = (600.0, 80.0, 80.0)
    fluid_mesh = box_mesh(shape, size)
    solid_mesh = box_mesh(shape, size, origin=(size[0], 0.0, 0.0))
    fluid, solid = discretize(fluid_mesh), discretize(solid_mesh)
    fluid_faces = tuple(face for face in fluid_mesh.boundary_faces if face.axis == 0 and face.side == 1)
    solid_faces = tuple(face for face in solid_mesh.boundary_faces if face.axis == 0 and face.side == 0)
    interface = planar_interface(fluid, fluid_faces, solid, solid_faces)

    water = AcousticMaterial(density=1000.0, vp=1500.0)
    # Zero lambda makes the sides traction-free for a planar longitudinal wave.
    solid_vp = 3000.0
    material = IsotropicMaterial(density=2500.0, vp=solid_vp, vs=solid_vp / np.sqrt(2))
    acoustic = acoustic_operator(fluid, fluid_mesh.region_ids, {0: water})
    elastic = elastic_operator(solid, solid_mesh.region_ids, {0: material})
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
