from dataclasses import dataclass, replace

import numpy as np

from .actuator import GimbalActuator
from .aero import AeroConfig, AeroState, compute_aerodynamics
from .controller import GimbalController
from .motor import load_thrust_curve
from .sensors import SensorSuite


@dataclass
class RocketParams:
    mass: float
    moment_arm: float
    pitch_moment_inertia: float


@dataclass
class RocketState:
    time: float = 0.0
    x: float = 0.0
    y: float = 0.0
    pitch: float = 0.0
    x_vel: float = 0.0
    y_vel: float = 0.0
    pitch_vel: float = 0.0
    x_accel: float = 0.0
    y_accel: float = 0.0
    pitch_accel: float = 0.0


@dataclass
class RocketInputs:
    thrust: float = 0.0
    gimbal_angle: float = 0.0
    commanded_gimbal_angle: float | None = None


class Rocket:
    def __init__(
        self,
        time_step,
        max_time,
        mass,
        moment_arm,
        moment_inertia,
        initial_angle,
        thrust_curve_filename,
        aero_config=None,
    ):
        self.max_time = max_time
        self.dt = time_step
        self.params = RocketParams(mass, moment_arm, moment_inertia)
        self.state = RocketState(pitch=initial_angle)
        self.inputs = RocketInputs()
        self.aero_config = aero_config or AeroConfig(enabled=False)
        self.aero_state = AeroState()
        self.actuator = GimbalActuator()
        self.controller = GimbalController()
        self.sensors = SensorSuite()
        self.history = []
        self.input_history = []
        self.aero_history = []
        self.actuator_history = []
        self.sensor_history = []
        self.estimate_history = []
        self.times, self.thrusts = self.import_thrust_curve(thrust_curve_filename)

    def import_thrust_curve(self, filename):
        return load_thrust_curve(filename, self.max_time, self.dt)

    def step(self, inputs=None):
        self.history.append(replace(self.state))
        sensor_readings = self.sensors.read(self.state)
        self.sensor_history.append(sensor_readings)

        if inputs is None:
            index = int(self.state.time / self.dt)
            thrust = self.thrusts[index]
            commanded_gimbal_angle = self.controller.get_gimbal_angle(
                sensor_readings,
                self.params,
                thrust,
                self.dt,
            )
            gimbal_angle = self.actuator.update(
                commanded_gimbal_angle,
                self.dt,
            )
            inputs = RocketInputs(
                thrust,
                gimbal_angle,
                commanded_gimbal_angle,
            )
        else:
            self.controller.update_estimate(sensor_readings, self.dt)

        self.estimate_history.append(self.controller.estimated_state)
        self.actuator_history.append(self.actuator.state)

        self.dynamics(inputs)
        self.input_history.append(replace(self.inputs))
        self.aero_history.append(self.aero_state)

        self.integrate()
        self.state.time += self.dt

    def dynamics(self, inputs):
        if not isinstance(inputs, RocketInputs):
            inputs = RocketInputs(inputs, self.inputs.gimbal_angle)
        elif inputs.commanded_gimbal_angle is None:
            inputs = replace(
                inputs,
                commanded_gimbal_angle=inputs.gimbal_angle,
            )

        self.inputs = inputs

        thrust_angle = self.state.pitch + inputs.gimbal_angle
        x_thrust = inputs.thrust * np.sin(thrust_angle)
        y_thrust = inputs.thrust * np.cos(thrust_angle)

        torque = (
            inputs.thrust
            * np.sin(inputs.gimbal_angle)
            * self.params.moment_arm
        )
        weight = 9.81 * self.params.mass
        if (
            self.state.y <= 0.0
            and self.state.y_vel <= 0.0
            and y_thrust <= weight
        ):
            self.aero_state = AeroState(wind_x_m_s=self.aero_config.wind_x_m_s)
        else:
            self.aero_state = compute_aerodynamics(
                self.aero_config,
                self.state.x_vel,
                self.state.y_vel,
                self.state.pitch,
                self.state.pitch_vel,
            )

        x_net = x_thrust + self.aero_state.x_force_n
        y_net = y_thrust + self.aero_state.y_force_n - weight
        pitch_torque = torque + self.aero_state.pitch_moment_n_m

        self.state.x_accel = x_net / self.params.mass
        self.state.y_accel = y_net / self.params.mass
        self.state.pitch_accel = (
            pitch_torque / self.params.pitch_moment_inertia
        )

    def integrate(self):
        next_x_vel = self.state.x_vel + self.state.x_accel * self.dt
        next_y_vel = self.state.y_vel + self.state.y_accel * self.dt
        next_pitch_vel = self.state.pitch_vel + (
            self.state.pitch_accel * self.dt
        )
        next_y = self.state.y + next_y_vel * self.dt

        if next_y <= 0.0:
            if self.state.y > 0.0:
                self.state.x += next_x_vel * self.dt
                self.state.pitch += next_pitch_vel * self.dt
            self.state.y = 0.0
            self.state.x_vel = 0.0
            self.state.y_vel = 0.0
            self.state.pitch_vel = 0.0
            return

        self.state.x_vel = next_x_vel
        self.state.y_vel = next_y_vel
        self.state.pitch_vel = next_pitch_vel
        self.state.y = next_y
        self.state.x += self.state.x_vel * self.dt
        self.state.pitch += self.state.pitch_vel * self.dt

    def run_sim(self):
        self.state.time = 0
        self.controller.reset_estimator(self.state)
        self.actuator.reset(0.0)
        while self.state.time < self.max_time:
            self.step()

    @property
    def time(self):
        return self.state.time

    @time.setter
    def time(self, value):
        self.state.time = value

    @property
    def mass(self):
        return self.params.mass

    @mass.setter
    def mass(self, value):
        self.params.mass = value

    @property
    def moment_arm(self):
        return self.params.moment_arm

    @moment_arm.setter
    def moment_arm(self, value):
        self.params.moment_arm = value

    @property
    def moment_inertia(self):
        return self.params.pitch_moment_inertia

    @moment_inertia.setter
    def moment_inertia(self, value):
        self.params.pitch_moment_inertia = value

    @property
    def x(self):
        return self.state.x

    @x.setter
    def x(self, value):
        self.state.x = value

    @property
    def y(self):
        return self.state.y

    @y.setter
    def y(self, value):
        self.state.y = value

    @property
    def theta(self):
        return self.state.pitch

    @theta.setter
    def theta(self, value):
        self.state.pitch = value

    @property
    def gimbal_theta(self):
        return self.inputs.gimbal_angle

    @property
    def x_vel(self):
        return self.state.x_vel

    @x_vel.setter
    def x_vel(self, value):
        self.state.x_vel = value

    @property
    def y_vel(self):
        return self.state.y_vel

    @y_vel.setter
    def y_vel(self, value):
        self.state.y_vel = value

    @property
    def theta_vel(self):
        return self.state.pitch_vel

    @theta_vel.setter
    def theta_vel(self, value):
        self.state.pitch_vel = value

    @property
    def x_accel(self):
        return self.state.x_accel

    @x_accel.setter
    def x_accel(self, value):
        self.state.x_accel = value

    @property
    def y_accel(self):
        return self.state.y_accel

    @y_accel.setter
    def y_accel(self, value):
        self.state.y_accel = value

    @property
    def theta_accel(self):
        return self.state.pitch_accel

    @theta_accel.setter
    def theta_accel(self, value):
        self.state.pitch_accel = value

    @property
    def x_arr(self):
        return [state.x for state in self.history]

    @property
    def y_arr(self):
        return [state.y for state in self.history]

    @property
    def theta_arr(self):
        return [state.pitch for state in self.history]
