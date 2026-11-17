import unittest

import numpy as np

from main import coupled_box
from my_specfem.coupling import planar_interface
from my_specfem.mesh import box_mesh
from my_specfem.sem import discretize


class CouplingTests(unittest.TestCase):
    def test_interface_loads_with_shared_edge(self):
        fluid_mesh = box_mesh((1, 2, 1), (2.0, 6.0, 4.0))
        solid_mesh = box_mesh((1, 2, 1), (3.0, 6.0, 4.0), origin=(2.0, 0.0, 0.0))
        fluid, solid = discretize(fluid_mesh), discretize(solid_mesh)
        fluid_faces = tuple(face for face in fluid_mesh.boundary_faces if face.axis == 0 and face.side == 1)
        solid_faces = tuple(face for face in solid_mesh.boundary_faces if face.axis == 0 and face.side == 0)
        interface = planar_interface(fluid, fluid_faces, solid, solid_faces[::-1])

        np.testing.assert_array_equal(interface.solid_normal, [-1, 0, 0])
        np.testing.assert_allclose(interface.surface_weights.sum(), 24.0)
        self.assertEqual(len(np.unique(interface.fluid_nodes)), 45)
        self.assertEqual(len(interface.fluid_nodes), 50)
        fluid_load = np.zeros((fluid.node_count, 1))
        solid_load = np.zeros((solid.node_count, 3))
        pressure = np.full_like(fluid_load, 3.0)
        displacement = np.broadcast_to([2.0, 5.0, -7.0], solid_load.shape)
        interface.add_fluid_load(fluid_load, displacement)
        interface.add_solid_load(solid_load, pressure)
        np.testing.assert_allclose(fluid_load.sum(), 48.0)
        np.testing.assert_allclose(solid_load.sum(axis=0), [72.0, 0.0, 0.0])

        rng = np.random.default_rng(42)
        pressure = rng.normal(size=fluid_load.shape)
        displacement = rng.normal(size=solid_load.shape)
        fluid_load.fill(0.0)
        solid_load.fill(0.0)
        interface.add_fluid_load(fluid_load, displacement)
        interface.add_solid_load(solid_load, pressure)
        np.testing.assert_allclose(np.sum(pressure * fluid_load), np.sum(displacement * solid_load))

        with self.assertRaisesRegex(ValueError, "same GLL degree"):
            planar_interface(fluid, fluid_faces, discretize(solid_mesh, degree=3), solid_faces)
        shifted = box_mesh((1, 2, 1), (3.0, 6.0, 4.0), origin=(2.1, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "coordinates must match"):
            planar_interface(fluid, fluid_faces, discretize(shifted), solid_faces)

    def test_plane_pulse_reflection_and_transmission(self):
        pressure, displacement = coupled_box()
        np.testing.assert_array_equal(pressure.time_s, displacement.time_s)
        self.assertEqual(tuple(channel.name for channel in pressure.channels), ("pressure",))
        self.assertEqual(tuple(channel.name for channel in displacement.channels), ("ux", "uy", "uz"))
        time = pressure.time_s
        fluid_speed, solid_speed = 1500.0, 3000.0
        fluid_impedance = 1000.0 * fluid_speed
        solid_impedance = 2500.0 * solid_speed
        reflection = (solid_impedance - fluid_impedance) / (solid_impedance + fluid_impedance)
        width = 35.0

        def travelling_pressure(offset):
            return (fluid_speed / width)**2 * (1 - offset**2) * np.exp(-0.5 * offset**2)

        incident = travelling_pressure((481 - 300 - fluid_speed * time) / width)
        reflected = reflection * travelling_pressure((2 * 600 - 481 - 300 - fluid_speed * time) / width)
        # p_transmitted = Z_s * du_x/dt for the right-going P wave.
        offset = (600 - 300 - fluid_speed * (time - (751 - 600) / solid_speed)) / width
        expected_u = (
            -(1 + reflection) * fluid_speed * offset * np.exp(-0.5 * offset**2)
            / (width * solid_impedance)
        )
        reflection_window = time > 0.21
        measured_reflection = pressure.values[:, 0, 0] - incident
        p_error = np.linalg.norm((measured_reflection - reflected)[reflection_window])
        p_error /= np.linalg.norm(reflected[reflection_window])
        u_error = np.linalg.norm(displacement.values[:, 0, 0] - expected_u)
        u_error /= np.linalg.norm(expected_u)
        self.assertLess(p_error, 0.02, f"relative reflected pressure error: {p_error:.3%}")
        self.assertLess(u_error, 0.02, f"relative transmitted displacement error: {u_error:.3%}")
        self.assertLess(np.max(np.abs(displacement.values[:, 0, 1:])), 1e-10)
