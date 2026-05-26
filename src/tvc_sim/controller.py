from dataclasses import dataclass
import math

from .estimator import ImuStateEstimator


@dataclass
class GimbalController:
    enabled: bool = True
    gimbal_angle: float = 0.0
    alpha_cmd: float = 0.0
    kp: float = 22.25
    kd: float = 3.124291252531
    max_gimbal_angle: float = 0.1745  # 10 deg
    min_control_thrust: float = 3.0

    def __post_init__(self):
        self.estimator = ImuStateEstimator()

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(value, high))

    def reset_estimator(self, state=None):
        self.estimator.reset(state)

    def update_estimate(self, readings, dt):
        return self.estimator.update(readings, dt)

    @property
    def estimated_state(self):
        return self.estimator.estimated_state

    def get_gimbal_angle(self, readings, params, thrust, dt):
        estimated_state = self.update_estimate(readings, dt)

        if (
            not self.enabled
            or thrust < self.min_control_thrust
            or params.moment_arm <= 0
        ):
            self.alpha_cmd = 0.0
            self.gimbal_angle = 0.0
            return self.gimbal_angle

        pitch_error = -estimated_state.pitch
        pitch_rate_error = -estimated_state.pitch_vel

        self.alpha_cmd = self.kp * pitch_error + self.kd * pitch_rate_error

        asin_input = (
            params.pitch_moment_inertia
            * self.alpha_cmd
            / (params.moment_arm * thrust)
        )
        asin_input = self.clamp(asin_input, -1, 1)

        gimbal_angle = math.asin(asin_input)
        self.gimbal_angle = self.clamp(
            gimbal_angle,
            -self.max_gimbal_angle,
            self.max_gimbal_angle,
        )
        return self.gimbal_angle
