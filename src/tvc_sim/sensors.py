from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorReadings:
    time: float
    pitch_vel: float
    x_accel: float
    y_accel: float


class Gyroscope:
    def __init__(self, noise_std=0.0, bias=0.0, rng=None):
        self.noise_std = noise_std
        self.bias = bias
        self.rng = rng

    def read(self, state):
        return state.pitch_vel + self.bias + self.noise()

    def noise(self):
        if self.noise_std <= 0.0 or self.rng is None:
            return 0.0
        return self.rng.normal(0.0, self.noise_std)


class Accelerometer:
    def __init__(
        self,
        noise_std=0.0,
        x_bias=0.0,
        y_bias=0.0,
        rng=None,
    ):
        self.noise_std = noise_std
        self.x_bias = x_bias
        self.y_bias = y_bias
        self.rng = rng

    def read(self, state):
        return (
            state.x_accel + self.x_bias + self.noise(),
            state.y_accel + self.y_bias + self.noise(),
        )

    def noise(self):
        if self.noise_std <= 0.0 or self.rng is None:
            return 0.0
        return self.rng.normal(0.0, self.noise_std)


class SensorSuite:
    def __init__(
        self,
        gyro_noise_std=0.0,
        gyro_bias=0.0,
        accel_noise_std=0.0,
        accel_x_bias=0.0,
        accel_y_bias=0.0,
        random_seed=0,
    ):
        rng = np.random.default_rng(random_seed)
        self.gyroscope = Gyroscope(gyro_noise_std, gyro_bias, rng)
        self.accelerometer = Accelerometer(
            accel_noise_std,
            accel_x_bias,
            accel_y_bias,
            rng,
        )

    def read(self, state):
        x_accel, y_accel = self.accelerometer.read(state)
        return SensorReadings(
            time=state.time,
            pitch_vel=self.gyroscope.read(state),
            x_accel=x_accel,
            y_accel=y_accel,
        )
