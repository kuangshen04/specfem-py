import unittest

import numpy as np

from main import coupled_box
from my_specfem.acoustic import AcousticMaterial, acoustic_operator
from my_specfem.coupling import fluid_solid_interface
from my_specfem.elastic import IsotropicMaterial, elastic_operator
from my_specfem.mesh import box_mesh
from my_specfem.sem import discretize, spectral_domain


class CouplingTests(unittest.TestCase):
    def test_interface_loads_with_shared_edge(self):
        mesh = box_mesh((2, 2, 1), (4.0, 6.0, 4.0), np.array([[[0], [0]], [[1], [1]]]))
        grid = discretize(mesh)
        fluid = spectral_domain(grid, np.array([0, 1]))
        solid = spectral_domain(grid, np.array([2, 3]))
        interface = fluid_solid_interface(fluid, solid)

        self.assertIs(fluid.grid, solid.grid)
        self.assertEqual(grid.node_count, 405)
        self.assertEqual(fluid.node_count, 225)
        self.assertEqual(solid.node_count, 225)
        np.testing.assert_array_equal(
            fluid.geometry_nodes[interface.fluid_nodes],
            solid.geometry_nodes[interface.solid_nodes],
        )
        np.testing.assert_array_equal(
            interface.solid_normals, np.broadcast_to([-1, 0, 0], (50, 3))
        )
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

        water = AcousticMaterial(1000.0, 1500.0)
        rock = IsotropicMaterial(2500.0, 3000.0, 1500.0)
        acoustic = acoustic_operator(fluid, {0: water})
        elastic = elastic_operator(solid, {1: rock})
        self.assertEqual(acoustic.mass.shape, (225,))
        self.assertEqual(elastic.mass.shape, (225,))
        self.assertTrue(np.all(acoustic.mass > 0))
        self.assertTrue(np.all(elastic.mass > 0))
        np.testing.assert_allclose(acoustic.mass.sum(), 48.0 / water.bulk_modulus)
        np.testing.assert_allclose(elastic.mass.sum(), 48.0 * rock.density)
        with self.assertRaisesRegex(ValueError, "outside this field domain"):
            fluid.locate((3.0, 1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "domain node count"):
            fluid.gather(np.zeros((grid.node_count, 1)))
        with self.assertRaisesRegex(ValueError, "domain elements and GLL axes"):
            fluid.assemble(grid.volume_weights)
        with self.assertRaisesRegex(ValueError, "share a spectral grid"):
            fluid_solid_interface(fluid, spectral_domain(discretize(mesh), np.array([2, 3])))
        with self.assertRaisesRegex(ValueError, "exactly one coupled domain"):
            fluid_solid_interface(fluid, spectral_domain(grid, np.array([1, 2, 3])))

    def test_stepped_interface_with_noncontiguous_domain(self):
        grid = discretize(box_mesh((2, 2, 1), (4.0, 6.0, 4.0)))
        fluid = spectral_domain(grid, np.array([3, 0, 1]))
        solid = spectral_domain(grid, np.array([2]))
        interface = fluid_solid_interface(fluid, solid)
        np.testing.assert_array_equal(
            fluid.geometry_nodes[interface.fluid_nodes],
            solid.geometry_nodes[interface.solid_nodes],
        )
        np.testing.assert_array_equal(
            np.unique(interface.solid_normals, axis=0), [[-1, 0, 0], [0, 1, 0]]
        )
        np.testing.assert_allclose(interface.surface_weights.sum(), 3.0 * 4.0 + 2.0 * 4.0)

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
