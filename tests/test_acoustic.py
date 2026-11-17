import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from main import acoustic_box
from my_specfem.acoustic import AcousticMaterial, acoustic_operator
from my_specfem.acquisition import AcousticPointSource, Receiver, Ricker
from my_specfem.forward import run_acoustic
from my_specfem.io import write_seismogram
from my_specfem.mesh import box_mesh
from my_specfem.sem import discretize, spectral_domain


class AcousticTests(unittest.TestCase):
    def test_lumped_mass_and_scalar_point_load(self):
        mesh = box_mesh((2, 1, 1), (120.0, 40.0, 30.0), np.array([[[0]], [[1]]]))
        domain = spectral_domain(discretize(mesh))
        materials = {0: AcousticMaterial(1000.0, 1500.0), 1: AcousticMaterial(1100.0, 1600.0)}
        operator = acoustic_operator(domain, materials)
        expected_mass = sum(60 * 40 * 30 / material.bulk_modulus for material in materials.values())
        np.testing.assert_allclose(operator.mass.sum(), expected_mass, rtol=1e-13)
        np.testing.assert_allclose(
            operator.stiffness(np.ones((domain.node_count, 1))), 0, atol=1e-14
        )

        stencil = domain.locate((23.0, 17.0, 12.0))
        source = AcousticPointSource(stencil, 0.01, Ricker(20.0))
        load = np.zeros((domain.node_count, 1))
        source.add_to(load, 0.0)
        np.testing.assert_allclose(load.sum(), 0.01)
        trace = run_acoustic(operator, source, (Receiver("A", stencil),), t0=0.0, dt=0.001, steps=1)
        self.assertLess(trace.values[0, 0, 0], 0)
        np.testing.assert_allclose(trace.values[0, 0], -stencil.sample(load / operator.mass[:, None]))

    def test_rigid_box_standing_wave(self):
        length = 120.0
        vp = 1500.0
        mesh = box_mesh((4, 1, 1), (length, 40.0, 30.0))
        domain = spectral_domain(discretize(mesh))
        operator = acoustic_operator(domain, {0: AcousticMaterial(1000.0, vp)})
        wave_number = np.pi / length
        omega = vp * wave_number
        potential = np.cos(wave_number * domain.coordinates[:, :1])
        position = (23.0, 17.0, 12.0)
        trace = run_acoustic(
            operator, None, (Receiver("A", domain.locate(position)),),
            t0=0.0, dt=0.00025, steps=640, initial_potential=potential,
        )
        # Neumann fundamental mode, with p = -d²chi/dt² and zero initial velocity.
        expected = omega**2 * np.cos(wave_number * position[0]) * np.cos(omega * trace.time_s)
        pressure = trace.values[:, 0, 0]
        self.assertGreater(pressure[0], 0)
        relative_error = np.linalg.norm(pressure - expected) / np.linalg.norm(expected)
        self.assertLess(relative_error, 0.01)

    def test_point_source_box_writes_pressure(self):
        trace = acoustic_box()
        self.assertEqual(trace.values.shape, (201, 2, 1))
        self.assertTrue(np.isfinite(trace.values).all())
        self.assertTrue(np.all(np.max(np.abs(trace.values), axis=(0, 2)) > 0))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "pressure.csv"
            write_seismogram(path, trace)
            with path.open(encoding="utf-8", newline="") as stream:
                header, *rows = csv.reader(stream)
        self.assertEqual(header, ["time_s", "station_id", "pressure_Pa"])
        self.assertEqual(len(rows), 402)
        self.assertEqual(float(rows[-1][-1]), trace.values[-1, -1, 0])
