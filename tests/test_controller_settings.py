import json
from pathlib import Path
import tempfile
import unittest

import tvc_sim_ui.streamlit_app as app


class ControllerSettingsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = app.CONTROLLER_SETTINGS_PATH
        self.original_aero_path = app.AERO_SETTINGS_PATH
        self.original_rocket_path = app.ROCKET_SETTINGS_PATH
        app.CONTROLLER_SETTINGS_PATH = (
            Path(self.temp_dir.name) / "controller_settings.json"
        )
        app.AERO_SETTINGS_PATH = (
            Path(self.temp_dir.name) / "aerodynamics_settings.json"
        )
        app.ROCKET_SETTINGS_PATH = (
            Path(self.temp_dir.name) / "rocket_settings.json"
        )

    def tearDown(self):
        app.CONTROLLER_SETTINGS_PATH = self.original_path
        app.AERO_SETTINGS_PATH = self.original_aero_path
        app.ROCKET_SETTINGS_PATH = self.original_rocket_path
        self.temp_dir.cleanup()

    def write_settings(self, data):
        app.CONTROLLER_SETTINGS_PATH.write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    def read_settings(self):
        return json.loads(
            app.CONTROLLER_SETTINGS_PATH.read_text(encoding="utf-8")
        )

    def write_aero_settings(self, data):
        app.AERO_SETTINGS_PATH.write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    def read_aero_settings(self):
        return json.loads(app.AERO_SETTINGS_PATH.read_text(encoding="utf-8"))

    def write_rocket_settings(self, data):
        app.ROCKET_SETTINGS_PATH.write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    def read_rocket_settings(self):
        return json.loads(app.ROCKET_SETTINGS_PATH.read_text(encoding="utf-8"))

    def test_loads_legacy_flat_settings_as_ideal_profile(self):
        self.write_settings({"gimbal_kp": 1.2, "gimbal_kd": 0.34})

        settings = app.load_controller_settings()

        self.assertEqual(
            settings[app.CONTROLLER_PROFILE_IDEAL],
            {"gimbal_kp": 1.2, "gimbal_kd": 0.34},
        )
        self.assertEqual(settings[app.CONTROLLER_PROFILE_ACTUATOR], {})
        self.assertEqual(settings[app.CONTROLLER_PROFILE_FULL], {})

    def test_controller_profile_selects_full_mode_for_everything_on(self):
        self.assertEqual(
            app.controller_profile(
                actuator_enabled=True,
                sensor_noise_enabled=True,
                aero_enabled=True,
            ),
            app.CONTROLLER_PROFILE_FULL,
        )
        self.assertEqual(
            app.controller_profile(
                actuator_enabled=True,
                sensor_noise_enabled=False,
                aero_enabled=True,
            ),
            app.CONTROLLER_PROFILE_ACTUATOR,
        )
        self.assertEqual(
            app.controller_profile(
                actuator_enabled=False,
                sensor_noise_enabled=True,
                aero_enabled=True,
            ),
            app.CONTROLLER_PROFILE_IDEAL,
        )

    def test_mode_effects_map_to_expected_simulation_features(self):
        self.assertEqual(
            app.mode_effects(app.SIM_MODE_IDEAL),
            {
                "gimbal_enabled": True,
                "actuator_enabled": False,
                "sensor_noise_enabled": False,
                "aero_enabled": False,
            },
        )
        self.assertEqual(
            app.mode_effects(app.SIM_MODE_NOISY),
            {
                "gimbal_enabled": True,
                "actuator_enabled": True,
                "sensor_noise_enabled": True,
                "aero_enabled": False,
            },
        )
        self.assertEqual(
            app.mode_effects(app.SIM_MODE_FULL),
            {
                "gimbal_enabled": True,
                "actuator_enabled": True,
                "sensor_noise_enabled": True,
                "aero_enabled": True,
            },
        )

    def test_controller_profile_for_mode(self):
        self.assertEqual(
            app.controller_profile_for_mode(app.SIM_MODE_IDEAL),
            app.CONTROLLER_PROFILE_IDEAL,
        )
        self.assertEqual(
            app.controller_profile_for_mode(app.SIM_MODE_NOISY),
            app.CONTROLLER_PROFILE_ACTUATOR,
        )
        self.assertEqual(
            app.controller_profile_for_mode(app.SIM_MODE_FULL),
            app.CONTROLLER_PROFILE_FULL,
        )

    def test_saving_actuator_profile_preserves_ideal_profile(self):
        self.write_settings(
            {
                app.CONTROLLER_PROFILE_IDEAL: {
                    "gimbal_kp": 0.1,
                    "gimbal_kd": 0.01,
                },
                app.CONTROLLER_PROFILE_ACTUATOR: {
                    "gimbal_kp": 1.0,
                    "gimbal_kd": 0.2,
                },
            }
        )

        app.save_controller_settings(
            app.CONTROLLER_PROFILE_ACTUATOR,
            2.0,
            0.4,
        )

        self.assertEqual(
            self.read_settings(),
            {
                app.CONTROLLER_PROFILE_IDEAL: {
                    "gimbal_kp": 0.1,
                    "gimbal_kd": 0.01,
                },
                app.CONTROLLER_PROFILE_ACTUATOR: {
                    "gimbal_kp": 2.0,
                    "gimbal_kd": 0.4,
                },
                app.CONTROLLER_PROFILE_FULL: {},
            },
        )

    def test_saving_full_profile_preserves_existing_profiles(self):
        self.write_settings(
            {
                app.CONTROLLER_PROFILE_IDEAL: {
                    "gimbal_kp": 0.1,
                    "gimbal_kd": 0.01,
                },
                app.CONTROLLER_PROFILE_ACTUATOR: {
                    "gimbal_kp": 4.0,
                    "gimbal_kd": 8.0,
                },
            }
        )

        app.save_controller_settings(
            app.CONTROLLER_PROFILE_FULL,
            22.25,
            3.124291252531,
        )

        self.assertEqual(
            self.read_settings(),
            {
                app.CONTROLLER_PROFILE_IDEAL: {
                    "gimbal_kp": 0.1,
                    "gimbal_kd": 0.01,
                },
                app.CONTROLLER_PROFILE_ACTUATOR: {
                    "gimbal_kp": 4.0,
                    "gimbal_kd": 8.0,
                },
                app.CONTROLLER_PROFILE_FULL: {
                    "gimbal_kp": 22.25,
                    "gimbal_kd": 3.124291252531,
                },
            },
        )

    def test_loads_nested_aero_settings(self):
        self.write_aero_settings(
            {
                app.AERO_SETTINGS_KEY: {
                    "aero_enabled": False,
                    "air_density_kg_m3": 1.1,
                    "drag_coefficient": 0.72,
                    "body_diameter_m": 0.054,
                    "normal_force_slope": 2.4,
                    "cp_cg_offset_m": 0.12,
                    "pitch_damping_coefficient": 0.03,
                    "wind_x_m_s": 4.0,
                }
            }
        )

        settings = app.load_aero_settings()

        self.assertEqual(
            settings,
            {
                "aero_enabled": False,
                "air_density_kg_m3": 1.1,
                "drag_coefficient": 0.72,
                "body_diameter_m": 0.054,
                "normal_force_slope": 2.4,
                "cp_cg_offset_m": 0.12,
                "pitch_damping_coefficient": 0.03,
                "wind_x_m_s": 4.0,
            },
        )

    def test_saving_aero_settings_preserves_existing_aero_data(self):
        self.write_aero_settings({"notes": "keep this"})

        app.save_aero_settings(
            True,
            1.225,
            0.6,
            0.05,
            2.0,
            0.1,
            0.02,
            3.0,
        )

        self.assertEqual(
            self.read_aero_settings(),
            {
                "notes": "keep this",
                app.AERO_SETTINGS_KEY: {
                    "aero_enabled": True,
                    "air_density_kg_m3": 1.225,
                    "drag_coefficient": 0.6,
                    "body_diameter_m": 0.05,
                    "normal_force_slope": 2.0,
                    "cp_cg_offset_m": 0.1,
                    "pitch_damping_coefficient": 0.02,
                    "wind_x_m_s": 3.0,
                },
            },
        )

    def test_saving_aero_settings_does_not_overwrite_controller_settings(self):
        controller_data = {
            app.CONTROLLER_PROFILE_IDEAL: {
                "gimbal_kp": 0.1,
                "gimbal_kd": 0.01,
            }
        }
        self.write_settings(controller_data)

        app.save_aero_settings(
            True,
            1.225,
            0.6,
            0.05,
            2.0,
            0.1,
            0.02,
            0.0,
        )

        self.assertEqual(self.read_settings(), controller_data)

    def test_loads_nested_rocket_settings(self):
        self.write_rocket_settings(
            {
                app.ROCKET_SETTINGS_KEY: {
                    "mass": 2.0,
                    "moment_arm": 0.367,
                    "pitch_moment_inertia": 0.24,
                }
            }
        )

        settings = app.load_rocket_settings()

        self.assertEqual(
            settings,
            {
                "mass": 2.0,
                "moment_arm": 0.367,
                "pitch_moment_inertia": 0.24,
            },
        )

    def test_saving_rocket_settings_preserves_existing_rocket_data(self):
        self.write_rocket_settings({"notes": "keep this"})

        app.save_rocket_settings(2.0, 0.367, 0.24)

        self.assertEqual(
            self.read_rocket_settings(),
            {
                "notes": "keep this",
                app.ROCKET_SETTINGS_KEY: {
                    "mass": 2.0,
                    "moment_arm": 0.367,
                    "pitch_moment_inertia": 0.24,
                },
            },
        )

    def test_saving_rocket_settings_does_not_overwrite_controller_settings(self):
        controller_data = {
            app.CONTROLLER_PROFILE_IDEAL: {
                "gimbal_kp": 0.1,
                "gimbal_kd": 0.01,
            }
        }
        self.write_settings(controller_data)

        app.save_rocket_settings(2.0, 0.367, 0.24)

        self.assertEqual(self.read_settings(), controller_data)


if __name__ == "__main__":
    unittest.main()
