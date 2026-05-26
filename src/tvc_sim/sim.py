from dataclasses import dataclass, field

import numpy as np

from .actuator import GimbalActuatorConfig
from .aero import AeroConfig
from .motor import default_thrust_curve_path
from .motor import load_thrust_curve_from_text
from .rocket import Rocket
from .sensors import SensorSuite


TIME_STEP = 0.005
MAX_TIME = 60
MOMENT_INERTIA = 0.24
MOMENT_ARM = 0.367
MASS = 2.0
INITIAL_ANGLE_DEG = 0
GIMBAL_KP = 22.25
GIMBAL_KD = 3.124291252531
GIMBAL_ENABLED = True
ACTUATOR_ENABLED = True
ACTUATOR_MAX_ANGLE_DEG = 8.0
ACTUATOR_SLEW_RATE_DEG_S = 500.0
ACTUATOR_DELAY_S = 0.02
ACTUATOR_LAG_S = 0.03
ACTUATOR_DEADBAND_DEG = 0.1
ACTUATOR_BIAS_DEG = 0.0
SENSOR_NOISE_ENABLED = True
SENSOR_RANDOM_SEED = 0
GYRO_NOISE_DEG_S = 0.5
GYRO_BIAS_DEG_S = 0.0
ACCEL_NOISE_M_S2 = 0.15
ACCEL_X_BIAS_M_S2 = 0.0
ACCEL_Y_BIAS_M_S2 = 0.0
AERO_ENABLED = True
AIR_DENSITY_KG_M3 = 1.225
DRAG_COEFFICIENT = 0.60
BODY_DIAMETER_M = 0.10
NORMAL_FORCE_SLOPE = 2.0
CP_CG_OFFSET_M = 0.146
PITCH_DAMPING_COEFFICIENT = 0.01
WIND_X_M_S = 0.0


@dataclass
class SimulationConfig:
    time_step: float = TIME_STEP
    max_time: float = MAX_TIME
    mass: float = MASS
    moment_arm: float = MOMENT_ARM
    pitch_moment_inertia: float = MOMENT_INERTIA
    initial_pitch_deg: float = INITIAL_ANGLE_DEG
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_x_vel: float = 0.0
    initial_y_vel: float = 0.0
    initial_pitch_rate_deg_s: float = 0.0
    gimbal_enabled: bool = GIMBAL_ENABLED
    gimbal_kp: float = GIMBAL_KP
    gimbal_kd: float = GIMBAL_KD
    actuator_enabled: bool = ACTUATOR_ENABLED
    actuator_max_angle_deg: float = ACTUATOR_MAX_ANGLE_DEG
    actuator_slew_rate_deg_s: float = ACTUATOR_SLEW_RATE_DEG_S
    actuator_delay_s: float = ACTUATOR_DELAY_S
    actuator_lag_s: float = ACTUATOR_LAG_S
    actuator_deadband_deg: float = ACTUATOR_DEADBAND_DEG
    actuator_bias_deg: float = ACTUATOR_BIAS_DEG
    sensor_noise_enabled: bool = SENSOR_NOISE_ENABLED
    sensor_random_seed: int = SENSOR_RANDOM_SEED
    gyro_noise_deg_s: float = GYRO_NOISE_DEG_S
    gyro_bias_deg_s: float = GYRO_BIAS_DEG_S
    accel_noise_m_s2: float = ACCEL_NOISE_M_S2
    accel_x_bias_m_s2: float = ACCEL_X_BIAS_M_S2
    accel_y_bias_m_s2: float = ACCEL_Y_BIAS_M_S2
    aero_enabled: bool = AERO_ENABLED
    air_density_kg_m3: float = AIR_DENSITY_KG_M3
    drag_coefficient: float = DRAG_COEFFICIENT
    body_diameter_m: float = BODY_DIAMETER_M
    normal_force_slope: float = NORMAL_FORCE_SLOPE
    cp_cg_offset_m: float = CP_CG_OFFSET_M
    pitch_damping_coefficient: float = PITCH_DAMPING_COEFFICIENT
    wind_x_m_s: float = WIND_X_M_S
    thrust_curve_filename: object = field(default_factory=default_thrust_curve_path)


def create_rocket(config=None):
    if config is None:
        config = SimulationConfig()

    rocket = Rocket(
        config.time_step,
        config.max_time,
        config.mass,
        config.moment_arm,
        config.pitch_moment_inertia,
        np.radians(config.initial_pitch_deg),
        config.thrust_curve_filename,
        AeroConfig(
            enabled=config.aero_enabled,
            air_density_kg_m3=config.air_density_kg_m3,
            drag_coefficient=config.drag_coefficient,
            body_diameter_m=config.body_diameter_m,
            normal_force_slope=config.normal_force_slope,
            cp_cg_offset_m=config.cp_cg_offset_m,
            pitch_damping_coefficient=config.pitch_damping_coefficient,
            wind_x_m_s=config.wind_x_m_s,
        ),
    )
    rocket.x = config.initial_x
    rocket.y = config.initial_y
    rocket.x_vel = config.initial_x_vel
    rocket.y_vel = config.initial_y_vel
    rocket.theta_vel = np.radians(config.initial_pitch_rate_deg_s)
    gyro_noise_deg_s = (
        config.gyro_noise_deg_s if config.sensor_noise_enabled else 0.0
    )
    accel_noise_m_s2 = (
        config.accel_noise_m_s2 if config.sensor_noise_enabled else 0.0
    )
    rocket.sensors = SensorSuite(
        gyro_noise_std=np.radians(gyro_noise_deg_s),
        gyro_bias=np.radians(config.gyro_bias_deg_s),
        accel_noise_std=accel_noise_m_s2,
        accel_x_bias=config.accel_x_bias_m_s2,
        accel_y_bias=config.accel_y_bias_m_s2,
        random_seed=config.sensor_random_seed,
    )
    rocket.controller.enabled = config.gimbal_enabled
    rocket.controller.kp = config.gimbal_kp
    rocket.controller.kd = config.gimbal_kd
    rocket.controller.max_gimbal_angle = np.radians(
        config.actuator_max_angle_deg
    )
    rocket.actuator.config = GimbalActuatorConfig(
        enabled=config.actuator_enabled,
        max_angle=np.radians(config.actuator_max_angle_deg),
        slew_rate=np.radians(config.actuator_slew_rate_deg_s),
        delay=config.actuator_delay_s,
        lag=config.actuator_lag_s,
        deadband=np.radians(config.actuator_deadband_deg),
        bias=np.radians(config.actuator_bias_deg),
    )
    rocket.actuator.reset(0.0)
    return rocket


def run_simulation(config=None, thrust_curve_text=None):
    rocket = create_rocket(config)
    if thrust_curve_text is not None:
        rocket.times, rocket.thrusts = load_thrust_curve_from_text(
            thrust_curve_text,
            rocket.max_time,
            rocket.dt,
        )
    rocket.run_sim()
    return rocket


def main():
    from .plotting import plot_trajectory

    plot_trajectory(run_simulation())
