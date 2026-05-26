from dataclasses import dataclass
import math


MAX_EFFECTIVE_ANGLE_OF_ATTACK_RAD = math.radians(30.0)


@dataclass(frozen=True)
class AeroConfig:
    enabled: bool = True
    air_density_kg_m3: float = 1.225
    drag_coefficient: float = 0.60
    body_diameter_m: float = 0.10
    normal_force_slope: float = 2.0
    cp_cg_offset_m: float = 0.146
    pitch_damping_coefficient: float = 0.01
    wind_x_m_s: float = 0.0

    @property
    def reference_area_m2(self):
        radius = self.body_diameter_m / 2.0
        return math.pi * radius * radius


@dataclass(frozen=True)
class AeroState:
    relative_x_vel_m_s: float = 0.0
    relative_y_vel_m_s: float = 0.0
    relative_speed_m_s: float = 0.0
    dynamic_pressure_pa: float = 0.0
    angle_of_attack_rad: float = 0.0
    drag_force_n: float = 0.0
    normal_force_n: float = 0.0
    x_force_n: float = 0.0
    y_force_n: float = 0.0
    restoring_pitch_moment_n_m: float = 0.0
    pitch_damping_moment_n_m: float = 0.0
    pitch_moment_n_m: float = 0.0
    wind_x_m_s: float = 0.0


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, low, high):
    return max(low, min(value, high))


def compute_aerodynamics(config, x_vel, y_vel, pitch, pitch_vel):
    if config is None or not config.enabled:
        return AeroState(wind_x_m_s=0.0 if config is None else config.wind_x_m_s)

    relative_x_vel = x_vel - config.wind_x_m_s
    relative_y_vel = y_vel
    speed = math.hypot(relative_x_vel, relative_y_vel)
    if not math.isfinite(speed):
        return AeroState(wind_x_m_s=config.wind_x_m_s)

    dynamic_pressure = 0.5 * config.air_density_kg_m3 * speed * speed

    if speed <= 0.0:
        damping_moment = -config.pitch_damping_coefficient * pitch_vel
        return AeroState(
            pitch_damping_moment_n_m=damping_moment,
            pitch_moment_n_m=damping_moment,
            wind_x_m_s=config.wind_x_m_s,
        )

    reference_area = config.reference_area_m2
    drag_force = dynamic_pressure * reference_area * config.drag_coefficient
    x_drag = -relative_x_vel / speed * drag_force
    y_drag = -relative_y_vel / speed * drag_force

    velocity_pitch = math.atan2(relative_x_vel, relative_y_vel)
    angle_of_attack = clamp(
        wrap_angle(pitch - velocity_pitch),
        -MAX_EFFECTIVE_ANGLE_OF_ATTACK_RAD,
        MAX_EFFECTIVE_ANGLE_OF_ATTACK_RAD,
    )
    normal_force = (
        dynamic_pressure
        * reference_area
        * config.normal_force_slope
        * angle_of_attack
    )

    normal_x = math.cos(velocity_pitch)
    normal_y = -math.sin(velocity_pitch)
    x_normal = -normal_x * normal_force
    y_normal = -normal_y * normal_force

    restoring_moment = -normal_force * config.cp_cg_offset_m
    damping_moment = -config.pitch_damping_coefficient * pitch_vel
    pitch_moment = restoring_moment + damping_moment

    return AeroState(
        relative_x_vel_m_s=relative_x_vel,
        relative_y_vel_m_s=relative_y_vel,
        relative_speed_m_s=speed,
        dynamic_pressure_pa=dynamic_pressure,
        angle_of_attack_rad=angle_of_attack,
        drag_force_n=drag_force,
        normal_force_n=normal_force,
        x_force_n=x_drag + x_normal,
        y_force_n=y_drag + y_normal,
        restoring_pitch_moment_n_m=restoring_moment,
        pitch_damping_moment_n_m=damping_moment,
        pitch_moment_n_m=pitch_moment,
        wind_x_m_s=config.wind_x_m_s,
    )
