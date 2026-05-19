import unittest

from tvc_sim.sim import SimulationConfig, run_simulation
from tvc_sim_ui.streamlit_app import build_history_frame, chart_y_domain


class SimulationSmokeTest(unittest.TestCase):
    def test_run_simulation_frame_has_actual_and_commanded_gimbal(self):
        config = SimulationConfig(
            time_step=0.01,
            max_time=0.03,
            sensor_noise_enabled=False,
            actuator_enabled=True,
            actuator_noise_deg=0.0,
        )

        rocket = run_simulation(config)
        frame = build_history_frame(rocket)

        self.assertIn("gimbal_deg", frame.columns)
        self.assertIn("commanded_gimbal_deg", frame.columns)
        self.assertGreater(len(frame), 0)

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
                    actuator_noise_deg=0.0,
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
            (-0.1, 0.1),
        )


if __name__ == "__main__":
    unittest.main()
