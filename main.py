from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from my_specfem.acoustic import AcousticMaterial, acoustic_operator
from my_specfem.acquisition import AcousticPointSource, PointForce, Receiver, Ricker, Seismogram
from my_specfem.elastic import IsotropicMaterial, elastic_operator
from my_specfem.forward import run_acoustic, run_elastic
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


def main() -> None:
    parser = ArgumentParser(description="Run a built-in 3D spectral-element example.")
    parser.add_argument("--case", choices=("elastic", "acoustic"), default="elastic")
    args = parser.parse_args()
    if args.case == "acoustic":
        filename, seismogram = "acoustic_seismogram.csv", acoustic_box()
    else:
        filename, seismogram = "seismogram.csv", homogeneous_box()
    path = Path("output") / filename
    path.parent.mkdir(exist_ok=True)
    write_seismogram(path, seismogram)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
