from pathlib import Path

import numpy as np

from my_specfem.acquisition import PointForce, Receiver, Ricker, Seismogram
from my_specfem.elastic import IsotropicMaterial, elastic_operator
from my_specfem.forward import run_elastic
from my_specfem.io import write_seismogram
from my_specfem.mesh import box_mesh
from my_specfem.sem import discretize


def homogeneous_box(*, dt: float = 0.001, steps: int = 200) -> Seismogram:
    mesh = box_mesh((4, 2, 2), (400.0, 200.0, 200.0))
    grid = discretize(mesh)
    materials = {0: IsotropicMaterial(density=2500.0, vp=3000.0, vs=1700.0)}
    operator = elastic_operator(grid, mesh.region_ids, materials)
    source = PointForce(
        grid.locate((87.0, 86.0, 87.0)), np.array([1.0e8, 0.0, 0.0]), Ricker(20.0)
    )
    receivers = (
        Receiver("STA01", grid.locate((143.0, 92.0, 88.0))),
        Receiver("STA02", grid.locate((269.0, 114.0, 93.0))),
    )
    return run_elastic(operator, source, receivers, t0=-0.075, dt=dt, steps=steps)


def main() -> None:
    path = Path("output") / "seismogram.csv"
    path.parent.mkdir(exist_ok=True)
    write_seismogram(path, homogeneous_box())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
