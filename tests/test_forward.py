import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from main import homogeneous_box
from my_specfem.acquisition import Channel, Seismogram
from my_specfem.elastic import IsotropicMaterial, elastic_operator
from my_specfem.io import write_seismogram
from my_specfem.mesh import box_mesh
from my_specfem.sem import discretize, spectral_domain
from my_specfem.time_integration import initialize, newmark_step


class SpectralElementTests(unittest.TestCase):
    def setUp(self):
        self.mesh = box_mesh((2, 1, 1), (8.0, 3.0, 2.0), np.array([[[0]], [[1]]]))
        self.domain = spectral_domain(discretize(self.mesh))
        self.materials = {
            0: IsotropicMaterial(2000.0, 2800.0, 1500.0),
            1: IsotropicMaterial(3000.0, 3500.0, 1900.0),
        }
        self.operator = elastic_operator(self.domain, self.materials)

    def test_shared_face_mass_assembly(self):
        mapping = self.domain.element_to_field_node
        self.assertEqual(len(self.mesh.boundary_faces), 10)
        self.assertEqual(self.domain.node_count, 225)
        shared_face = mapping[0, -1]
        np.testing.assert_array_equal(shared_face, mapping[1, 0])
        expected = (
            self.materials[0].density * self.domain.volume_weights[0, -1]
            + self.materials[1].density * self.domain.volume_weights[1, 0]
        )
        np.testing.assert_allclose(self.operator.mass[shared_face], expected)
        self.assertAlmostEqual(self.operator.mass.sum(), (2000.0 + 3000.0) * 24.0)

    def test_gradient_and_elastic_energy(self):
        coordinates = self.domain.coordinates
        # A non-symmetric displacement gradient checks both component and derivative axes.
        gradient = np.array([[0.5, 0.2, 0.3], [0.1, 0.4, -0.2], [0.0, 0.1, -0.1]])
        displacement = coordinates @ gradient.T
        actual = self.domain.gradient(self.domain.gather(displacement))
        np.testing.assert_allclose(actual, np.broadcast_to(gradient, actual.shape), atol=1e-13)
        stiffness = self.operator.stiffness(displacement)
        rigid = self.operator.stiffness(np.ones_like(displacement))
        self.assertLess(np.linalg.norm(rigid), 1e-13 * np.linalg.norm(stiffness))

        strain = 0.5 * (gradient + gradient.T)
        expected = sum(
            24.0
            * (
                material.lame_lambda * np.trace(strain) ** 2
                + 2 * material.shear_modulus * np.sum(strain**2)
            )
            for material in self.materials.values()
        )
        np.testing.assert_allclose(np.sum(displacement * stiffness), expected, rtol=1e-12)

    def test_point_load_and_receiver_pairing(self):
        position = (5.1, 0.7, 1.6)
        stencil = self.domain.locate(position)
        np.testing.assert_allclose(stencil.sample(self.domain.coordinates), position)
        u = np.random.default_rng(42).normal(size=(self.domain.node_count, 3))
        point_force = np.array([3.0, -2.0, 1.0])
        load = np.zeros_like(u)
        stencil.add_load(load, point_force)
        np.testing.assert_allclose(np.sum(u * load), stencil.sample(u) @ point_force)
        np.testing.assert_allclose(load.sum(axis=0), point_force)


class ForwardTests(unittest.TestCase):
    def test_newmark_oscillator(self):
        def acceleration(q, _time):
            return -4 * q

        errors = []
        for steps in (50, 100):
            state = initialize(0.0, np.array([1.0]), np.array([0.0]), acceleration)
            np.testing.assert_array_equal(state.qddot, [-4.0])
            for step in range(1, steps + 1):
                state = newmark_step(state, step / steps, acceleration)
            self.assertEqual(state.time, 1.0)
            errors.append(abs(state.q[0] - np.cos(2)))
        self.assertGreater(errors[0] / errors[1], 3.9)

    def test_homogeneous_seismogram_and_refinement(self):
        coarse = homogeneous_box()
        fine = homogeneous_box(dt=0.0005, steps=400)
        self.assertEqual(coarse.values.shape, (201, 2, 3))
        np.testing.assert_array_equal(coarse.time_s, fine.time_s[::2])
        np.testing.assert_array_equal(coarse.values[0], np.zeros((2, 3)))
        self.assertTrue(np.isfinite(coarse.values).all())
        self.assertTrue(np.all(np.max(np.abs(coarse.values), axis=(0, 2)) > 0))
        relative_error = (
            np.linalg.norm(coarse.values - fine.values[::2]) / np.linalg.norm(fine.values)
        )
        self.assertLess(relative_error, 0.02)

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.csv"
            write_seismogram(path, coarse)
            with path.open(encoding="utf-8", newline="") as stream:
                header, *rows = csv.reader(stream)
            self.assertEqual(header, ["time_s", "station_id", "ux_m", "uy_m", "uz_m"])
            self.assertEqual(len(rows), 402)
            self.assertEqual(rows[-1][1], "STA02")
            np.testing.assert_array_equal(
                [float(rows[-1][0]), *map(float, rows[-1][2:])],
                [coarse.time_s[-1], *coarse.values[-1, -1]],
            )

    def test_writer_uses_observable_channels(self):
        scalar = Seismogram(
            np.array([0.0]), ("STA01",), (Channel("pressure", "Pa"),), np.array([[[2.5]]])
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.csv"
            write_seismogram(path, scalar)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["time_s,station_id,pressure_Pa", "0.0,STA01,2.5"],
            )
