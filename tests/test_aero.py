import math
import unittest

from tvc_sim.aero import AeroConfig, compute_aerodynamics


class AerodynamicsTest(unittest.TestCase):
    def test_zero_speed_produces_zero_force_and_moment(self):
        aero = compute_aerodynamics(
            AeroConfig(),
            x_vel=0.0,
            y_vel=0.0,
            pitch=0.0,
            pitch_vel=0.0,
        )

        self.assertEqual(aero.dynamic_pressure_pa, 0.0)
        self.assertEqual(aero.drag_force_n, 0.0)
        self.assertEqual(aero.normal_force_n, 0.0)
        self.assertEqual(aero.pitch_moment_n_m, 0.0)

    def test_drag_opposes_air_relative_velocity(self):
        aero = compute_aerodynamics(
            AeroConfig(normal_force_slope=0.0),
            x_vel=10.0,
            y_vel=0.0,
            pitch=math.pi / 2.0,
            pitch_vel=0.0,
        )

        self.assertLess(aero.x_force_n, 0.0)
        self.assertAlmostEqual(aero.y_force_n, 0.0)

    def test_real_rocket_default_drag_matches_expected_scale(self):
        aero = compute_aerodynamics(
            AeroConfig(),
            x_vel=0.0,
            y_vel=50.0,
            pitch=0.0,
            pitch_vel=0.0,
        )

        self.assertAlmostEqual(aero.drag_force_n, 7.216, places=3)

    def test_testing_drag_example_matches_expected_scale(self):
        aero = compute_aerodynamics(
            AeroConfig(body_diameter_m=0.05),
            x_vel=0.0,
            y_vel=50.0,
            pitch=0.0,
            pitch_vel=0.0,
        )

        self.assertAlmostEqual(aero.drag_force_n, 1.804, places=3)

    def test_wind_changes_air_relative_velocity(self):
        aero = compute_aerodynamics(
            AeroConfig(wind_x_m_s=5.0),
            x_vel=0.0,
            y_vel=0.0,
            pitch=0.0,
            pitch_vel=0.0,
        )

        self.assertEqual(aero.relative_x_vel_m_s, -5.0)
        self.assertEqual(aero.relative_speed_m_s, 5.0)

    def test_angle_of_attack_creates_restoring_moment(self):
        aero = compute_aerodynamics(
            AeroConfig(),
            x_vel=0.0,
            y_vel=50.0,
            pitch=math.radians(10.0),
            pitch_vel=0.0,
        )

        self.assertGreater(aero.angle_of_attack_rad, 0.0)
        self.assertGreater(aero.normal_force_n, 0.0)
        self.assertLess(aero.restoring_pitch_moment_n_m, 0.0)

    def test_aerodynamic_force_does_not_add_energy(self):
        aero = compute_aerodynamics(
            AeroConfig(),
            x_vel=25.0,
            y_vel=50.0,
            pitch=math.radians(45.0),
            pitch_vel=0.0,
        )

        aero_power = (
            aero.x_force_n * aero.relative_x_vel_m_s
            + aero.y_force_n * aero.relative_y_vel_m_s
        )
        self.assertLessEqual(aero_power, 0.0)

    def test_positive_pitch_rate_creates_negative_damping_moment(self):
        aero = compute_aerodynamics(
            AeroConfig(pitch_damping_coefficient=0.02),
            x_vel=0.0,
            y_vel=0.0,
            pitch=0.0,
            pitch_vel=2.0,
        )

        self.assertEqual(aero.pitch_damping_moment_n_m, -0.04)


if __name__ == "__main__":
    unittest.main()
