import math
import unittest

from tvc_sim.actuator import GimbalActuator, GimbalActuatorConfig


class GimbalActuatorTest(unittest.TestCase):
    def test_disabled_returns_command_exactly(self):
        actuator = GimbalActuator(GimbalActuatorConfig(enabled=False))

        output = actuator.update(math.radians(25.0), 0.01)

        self.assertEqual(output, math.radians(25.0))

    def test_max_angle_clamps_output(self):
        actuator = GimbalActuator(
            GimbalActuatorConfig(
                enabled=True,
                max_angle=math.radians(10.0),
                slew_rate=math.radians(1000.0),
                delay=0.0,
                lag=0.0,
                deadband=0.0,
                noise_std=0.0,
            )
        )

        output = actuator.update(math.radians(20.0), 0.01)

        self.assertAlmostEqual(output, math.radians(10.0))

    def test_slew_rate_limits_output_change(self):
        actuator = GimbalActuator(
            GimbalActuatorConfig(
                enabled=True,
                max_angle=math.radians(30.0),
                slew_rate=math.radians(100.0),
                delay=0.0,
                lag=0.0,
                deadband=0.0,
                noise_std=0.0,
            )
        )

        output = actuator.update(math.radians(20.0), 0.01)

        self.assertAlmostEqual(output, math.radians(1.0))

    def test_delay_holds_command_for_configured_steps(self):
        actuator = GimbalActuator(
            GimbalActuatorConfig(
                enabled=True,
                max_angle=math.radians(30.0),
                slew_rate=math.radians(1000.0),
                delay=0.02,
                lag=0.0,
                deadband=0.0,
                noise_std=0.0,
            )
        )

        outputs = [
            actuator.update(math.radians(10.0), 0.01),
            actuator.update(math.radians(10.0), 0.01),
            actuator.update(math.radians(10.0), 0.01),
        ]

        self.assertAlmostEqual(outputs[0], 0.0)
        self.assertAlmostEqual(outputs[1], 0.0)
        self.assertAlmostEqual(outputs[2], math.radians(10.0))

    def test_lag_approaches_target_without_instant_jump(self):
        actuator = GimbalActuator(
            GimbalActuatorConfig(
                enabled=True,
                max_angle=math.radians(30.0),
                slew_rate=math.radians(1000.0),
                delay=0.0,
                lag=0.1,
                deadband=0.0,
                noise_std=0.0,
            )
        )

        first = actuator.update(math.radians(10.0), 0.01)
        second = actuator.update(math.radians(10.0), 0.01)

        self.assertGreater(first, 0.0)
        self.assertLess(first, math.radians(10.0))
        self.assertGreater(second, first)
        self.assertLess(second, math.radians(10.0))

    def test_deadband_holds_small_changes(self):
        actuator = GimbalActuator(
            GimbalActuatorConfig(
                enabled=True,
                max_angle=math.radians(30.0),
                slew_rate=math.radians(1000.0),
                delay=0.0,
                lag=0.0,
                deadband=math.radians(0.5),
                noise_std=0.0,
            )
        )

        output = actuator.update(math.radians(0.25), 0.01)

        self.assertAlmostEqual(output, 0.0)


if __name__ == "__main__":
    unittest.main()
