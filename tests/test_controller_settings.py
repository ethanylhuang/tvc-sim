import json
from pathlib import Path
import tempfile
import unittest

import tvc_sim_ui.streamlit_app as app


class ControllerSettingsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = app.CONTROLLER_SETTINGS_PATH
        app.CONTROLLER_SETTINGS_PATH = (
            Path(self.temp_dir.name) / "controller_settings.json"
        )

    def tearDown(self):
        app.CONTROLLER_SETTINGS_PATH = self.original_path
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

    def test_loads_legacy_flat_settings_as_ideal_profile(self):
        self.write_settings({"gimbal_kp": 1.2, "gimbal_kd": 0.34})

        settings = app.load_controller_settings()

        self.assertEqual(
            settings[app.CONTROLLER_PROFILE_IDEAL],
            {"gimbal_kp": 1.2, "gimbal_kd": 0.34},
        )
        self.assertEqual(settings[app.CONTROLLER_PROFILE_ACTUATOR], {})

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
            },
        )


if __name__ == "__main__":
    unittest.main()
