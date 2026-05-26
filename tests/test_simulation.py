import math
import unittest

import pandas as pd

from tvc_sim.rocket import RocketInputs
from tvc_sim.motor import DEFAULT_THRUST_CURVE, default_thrust_curve_path
from tvc_sim.sim import SimulationConfig, run_simulation
from tvc_sim.sim import create_rocket
from tvc_sim_ui.streamlit_app import (
    build_history_frame,
    chart_y_domain,
    thrust_to_weight_summary,
)


class SimulationSmokeTest(unittest.TestCase):
    def test_default_motor_is_h100w(self):
        self.assertEqual(DEFAULT_THRUST_CURVE, "AeroTech_H100W_DMS.csv")
        self.assertEqual(
            default_thrust_curve_path().name,
            "AeroTech_H100W_DMS.csv",
        )

    def test_default_constants_match_real_rocket_baseline(self):
        config = SimulationConfig()

        self.assertEqual(config.mass, 2.0)
        self.assertEqual(config.moment_arm, 0.367)
        self.assertEqual(config.pitch_moment_inertia, 0.24)
        self.assertEqual(config.initial_pitch_deg, 0)
        self.assertEqual(config.body_diameter_m, 0.10)
        self.assertEqual(config.cp_cg_offset_m, 0.146)
        self.assertEqual(config.gimbal_kp, 22.25)
        self.assertEqual(config.gimbal_kd, 3.124291252531)
        self.assertTrue(config.actuator_enabled)

    def test_run_simulation_frame_has_actual_and_commanded_gimbal(self):
        config = SimulationConfig(
            time_step=0.01,
            max_time=0.03,
            sensor_noise_enabled=False,
            actuator_enabled=True,
        )

        rocket = run_simulation(config)
        frame = build_history_frame(rocket)

        self.assertIn("gimbal_deg", frame.columns)
        self.assertIn("commanded_gimbal_deg", frame.columns)
        self.assertGreater(len(frame), 0)

    def test_gimbaled_thrust_changes_translation_direction(self):
        rocket = create_rocket(
            SimulationConfig(
                sensor_noise_enabled=False,
                aero_enabled=False,
                initial_pitch_deg=0.0,
            )
        )

        rocket.dynamics(
            RocketInputs(
                thrust=10.0,
                gimbal_angle=math.radians(5.0),
            )
        )

        self.assertGreater(rocket.x_accel, 0.0)
        self.assertAlmostEqual(
            rocket.x_accel,
            10.0 * math.sin(math.radians(5.0)) / rocket.mass,
        )
        self.assertAlmostEqual(
            rocket.y_accel,
            (10.0 * math.cos(math.radians(5.0)) - 9.81 * rocket.mass)
            / rocket.mass,
        )

    def test_controller_gimbal_limit_matches_physical_actuator_limit(self):
        rocket = create_rocket(SimulationConfig(actuator_max_angle_deg=8.0))

        self.assertAlmostEqual(
            rocket.controller.max_gimbal_angle,
            math.radians(8.0),
        )

    def test_ground_contact_does_not_accumulate_downward_velocity(self):
        rocket = create_rocket(
            SimulationConfig(
                aero_enabled=False,
                gimbal_enabled=False,
                sensor_noise_enabled=False,
            )
        )

        for _ in range(10):
            rocket.step(RocketInputs(thrust=0.0, gimbal_angle=0.0))

        self.assertEqual(rocket.y, 0.0)
        self.assertEqual(rocket.y_vel, 0.0)
        self.assertEqual(rocket.theta_vel, 0.0)

    def test_ground_contact_holds_low_thrust_gimbaled_rocket(self):
        rocket = create_rocket(
            SimulationConfig(
                aero_enabled=False,
                sensor_noise_enabled=False,
                initial_pitch_deg=0.0,
            )
        )

        rocket.step(
            RocketInputs(
                thrust=rocket.mass * 9.81 * 0.5,
                gimbal_angle=math.radians(5.0),
            )
        )

        self.assertEqual(rocket.y, 0.0)
        self.assertEqual(rocket.x_vel, 0.0)
        self.assertEqual(rocket.y_vel, 0.0)
        self.assertEqual(rocket.theta_vel, 0.0)

    def test_gimbal_chart_domain_depends_on_actuator_imperfections(self):
        ideal_frame = build_history_frame(
            run_simulation(
                SimulationConfig(
                    time_step=0.01,
                    max_time=0.03,
                    actuator_enabled=False,
                    sensor_noise_enabled=False,
                )
            )
        )
        actuator_frame = build_history_frame(
            run_simulation(
                SimulationConfig(
                    time_step=0.01,
                    max_time=0.03,
                    actuator_enabled=True,
                    sensor_noise_enabled=False,
                )
            )
        )

        self.assertEqual(
            chart_y_domain(
                ideal_frame,
                ["commanded_gimbal_deg", "gimbal_deg"],
                "gimbal_deg",
            ),
            (-0.02, 0.02),
        )
        self.assertEqual(
            chart_y_domain(
                actuator_frame,
                ["commanded_gimbal_deg", "gimbal_deg"],
                "gimbal_deg",
            ),
            (-0.2, 0.2),
        )

    def test_thrust_to_weight_summary_uses_selected_mass(self):
        frame = pd.DataFrame({"thrust_n": [0.0, 10.0, 20.0]})

        summary = thrust_to_weight_summary(frame, mass=2.0)

        self.assertAlmostEqual(summary["max"], 20.0 / (2.0 * 9.81))
        self.assertAlmostEqual(summary["average"], 15.0 / (2.0 * 9.81))

    def test_zero_gain_actuator_imperfections_do_not_move_pitch(self):
        config = SimulationConfig(
            gimbal_kp=0.0,
            gimbal_kd=0.0,
            actuator_enabled=True,
            sensor_noise_enabled=False,
            initial_pitch_deg=10.0,
            aero_enabled=False,
        )

        frame = build_history_frame(run_simulation(config))

        self.assertEqual(frame["commanded_gimbal_deg"].abs().max(), 0.0)
        self.assertEqual(frame["gimbal_deg"].abs().max(), 0.0)
        self.assertEqual(frame["pitch_deg"].min(), 10.0)
        self.assertEqual(frame["pitch_deg"].max(), 10.0)

    def test_aero_outputs_are_in_history_frame(self):
        frame = build_history_frame(
            run_simulation(
                SimulationConfig(
                    time_step=0.01,
                    max_time=0.03,
                    sensor_noise_enabled=False,
                )
            )
        )

        for column in (
            "dynamic_pressure_pa",
            "drag_n",
            "angle_of_attack_deg",
            "normal_force_n",
            "aero_pitch_moment_n_m",
            "wind_x_m_s",
        ):
            self.assertIn(column, frame.columns)

    def test_default_aero_simulation_outputs_are_finite(self):
        frame = build_history_frame(run_simulation())

        for column in (
            "x_m",
            "y_m",
            "pitch_deg",
            "dynamic_pressure_pa",
            "drag_n",
            "angle_of_attack_deg",
            "aero_pitch_moment_n_m",
        ):
            self.assertTrue(frame[column].map(math.isfinite).all(), column)

    def test_aero_disabled_preserves_no_aero_pitch_behavior(self):
        frame = build_history_frame(
            run_simulation(
                SimulationConfig(
                    gimbal_enabled=False,
                    sensor_noise_enabled=False,
                    initial_pitch_deg=10.0,
                    aero_enabled=False,
                )
            )
        )

        self.assertEqual(frame["pitch_deg"].min(), 10.0)
        self.assertEqual(frame["pitch_deg"].max(), 10.0)

    def test_drag_reduces_apogee_and_max_speed(self):
        base_config = {
            "gimbal_enabled": False,
            "sensor_noise_enabled": False,
            "initial_pitch_deg": 0.0,
            "max_time": 3.0,
        }
        no_aero = build_history_frame(
            run_simulation(SimulationConfig(**base_config, aero_enabled=False))
        )
        with_aero = build_history_frame(
            run_simulation(SimulationConfig(**base_config, aero_enabled=True))
        )

        no_aero_speed = (
            no_aero["x_vel_m_s"] ** 2 + no_aero["y_vel_m_s"] ** 2
        ) ** 0.5
        with_aero_speed = (
            with_aero["x_vel_m_s"] ** 2 + with_aero["y_vel_m_s"] ** 2
        ) ** 0.5

        self.assertLess(with_aero["y_m"].max(), no_aero["y_m"].max())
        self.assertLess(with_aero_speed.max(), no_aero_speed.max())

    def test_initial_pitch_does_not_increase_full_sim_apogee(self):
        base_config = {
            "gimbal_enabled": True,
            "actuator_enabled": True,
            "sensor_noise_enabled": False,
            "aero_enabled": True,
            "gimbal_kp": 22.25,
            "gimbal_kd": 3.124291252531,
            "max_time": 20.0,
        }
        vertical = build_history_frame(
            run_simulation(
                SimulationConfig(**base_config, initial_pitch_deg=0.0)
            )
        )
        tilted = build_history_frame(
            run_simulation(
                SimulationConfig(**base_config, initial_pitch_deg=30.0)
            )
        )

        self.assertLess(tilted["y_m"].max(), vertical["y_m"].max())

    def test_horizontal_wind_changes_range(self):
        base_config = {
            "gimbal_enabled": False,
            "sensor_noise_enabled": False,
            "initial_pitch_deg": 0.0,
            "aero_enabled": True,
            "max_time": 1.0,
        }
        no_wind = build_history_frame(
            run_simulation(SimulationConfig(**base_config, wind_x_m_s=0.0))
        )
        with_wind = build_history_frame(
            run_simulation(SimulationConfig(**base_config, wind_x_m_s=5.0))
        )

        self.assertNotEqual(with_wind["x_m"].iloc[-1], no_wind["x_m"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
