from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GimbalActuatorConfig:
    enabled: bool = False
    max_angle: float = math.radians(8.0)
    slew_rate: float = math.radians(500.0)
    delay: float = 0.02
    lag: float = 0.03
    deadband: float = math.radians(0.1)
    bias: float = 0.0
    noise_std: float = math.radians(0.02)
    random_seed: int = 1


@dataclass(frozen=True)
class GimbalActuatorState:
    commanded_angle: float = 0.0
    delayed_command_angle: float = 0.0
    position_angle: float = 0.0
    output_angle: float = 0.0


class GimbalActuator:
    def __init__(self, config=None):
        self.config = config or GimbalActuatorConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.state = GimbalActuatorState()
        self._delay_buffer = deque()
        self._delay_steps = None

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(value, high))

    def reset(self, angle=0.0):
        self.rng = np.random.default_rng(self.config.random_seed)
        self.state = GimbalActuatorState(
            commanded_angle=angle,
            delayed_command_angle=angle,
            position_angle=angle,
            output_angle=angle,
        )
        self._delay_buffer.clear()
        self._delay_steps = None

    def update(self, commanded_angle, dt):
        if not self.config.enabled:
            self.state = GimbalActuatorState(
                commanded_angle=commanded_angle,
                delayed_command_angle=commanded_angle,
                position_angle=commanded_angle,
                output_angle=commanded_angle,
            )
            return commanded_angle

        max_angle = abs(self.config.max_angle)
        command = self.clamp(commanded_angle, -max_angle, max_angle)
        delayed_command = self._apply_delay(command, dt)
        target = self._apply_deadband(delayed_command)
        target = self._apply_slew_limit(target, dt)
        position = self._apply_lag(target, dt)
        output = position + self.config.bias + self._noise()
        output = self.clamp(output, -max_angle, max_angle)

        self.state = GimbalActuatorState(
            commanded_angle=commanded_angle,
            delayed_command_angle=delayed_command,
            position_angle=position,
            output_angle=output,
        )
        return output

    def _apply_delay(self, command, dt):
        delay_steps = 0
        if dt > 0.0 and self.config.delay > 0.0:
            delay_steps = max(0, math.ceil(self.config.delay / dt))

        if delay_steps == 0:
            self._delay_buffer.clear()
            self._delay_steps = 0
            return command

        if self._delay_steps != delay_steps:
            self._delay_buffer = deque(
                [self.state.delayed_command_angle] * delay_steps
            )
            self._delay_steps = delay_steps

        self._delay_buffer.append(command)
        return self._delay_buffer.popleft()

    def _apply_deadband(self, target):
        if abs(target - self.state.position_angle) < abs(self.config.deadband):
            return self.state.position_angle
        return target

    def _apply_slew_limit(self, target, dt):
        if dt <= 0.0 or self.config.slew_rate <= 0.0:
            return self.state.position_angle

        max_step = abs(self.config.slew_rate) * dt
        delta = target - self.state.position_angle
        delta = self.clamp(delta, -max_step, max_step)
        return self.state.position_angle + delta

    def _apply_lag(self, target, dt):
        if dt <= 0.0 or self.config.lag <= 0.0:
            return target

        alpha = 1.0 - math.exp(-dt / self.config.lag)
        return self.state.position_angle + alpha * (
            target - self.state.position_angle
        )

    def _noise(self):
        if self.config.noise_std <= 0.0:
            return 0.0
        return self.rng.normal(0.0, self.config.noise_std)
