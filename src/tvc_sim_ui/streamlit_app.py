import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from tvc_sim.sim import SimulationConfig, run_simulation


CONFIG_FIELDS = (
    "time_step",
    "max_time",
    "mass",
    "moment_arm",
    "pitch_moment_inertia",
    "initial_pitch_deg",
    "gimbal_enabled",
    "gimbal_kp",
    "gimbal_kd",
    "actuator_enabled",
    "actuator_max_angle_deg",
    "actuator_slew_rate_deg_s",
    "actuator_delay_s",
    "actuator_lag_s",
    "actuator_deadband_deg",
    "actuator_bias_deg",
    "sensor_noise_enabled",
    "sensor_random_seed",
    "gyro_noise_deg_s",
    "accel_noise_m_s2",
    "aero_enabled",
    "air_density_kg_m3",
    "drag_coefficient",
    "body_diameter_m",
    "normal_force_slope",
    "cp_cg_offset_m",
    "pitch_damping_coefficient",
    "wind_x_m_s",
)
MAX_CHART_POINTS = 1200
DEFAULT_DATA_ROWS = 500
HISTORY_FRAME_VERSION = 15
CHART_VALUE_COLUMN = "__chart_value"
CHART_DOMAIN_PADDING = 0.05
BURNOUT_THRUST_THRESHOLD = 0.0
GIMBAL_DOMAIN_IDEAL = (-0.02, 0.02)
GIMBAL_DOMAIN_ACTUATOR = (-0.2, 0.2)
CONTROLLER_SETTINGS_PATH = (
    Path.home() / ".tvc_sim" / "controller_settings.json"
)
AERO_SETTINGS_PATH = Path.home() / ".tvc_sim" / "aerodynamics_settings.json"
AERO_SETTINGS_KEY = "aerodynamics"
ROCKET_SETTINGS_PATH = Path.home() / ".tvc_sim" / "rocket_settings.json"
ROCKET_SETTINGS_KEY = "rocket"
SIM_MODE_IDEAL = "ideal"
SIM_MODE_NOISY = "noisy"
SIM_MODE_FULL = "full"
SIM_MODES = (
    SIM_MODE_IDEAL,
    SIM_MODE_NOISY,
    SIM_MODE_FULL,
)
SIM_MODE_LABELS = {
    SIM_MODE_IDEAL: "Ideal",
    SIM_MODE_NOISY: "Noisy",
    SIM_MODE_FULL: "Full",
}
SIM_MODE_BY_LABEL = {
    label: mode for mode, label in SIM_MODE_LABELS.items()
}
DEFAULT_SIM_MODE = SIM_MODE_FULL
CONTROLLER_PROFILE_IDEAL = "ideal"
CONTROLLER_PROFILE_ACTUATOR = "actuator"
CONTROLLER_PROFILE_FULL = "full"
CONTROLLER_PROFILES = (
    CONTROLLER_PROFILE_IDEAL,
    CONTROLLER_PROFILE_ACTUATOR,
    CONTROLLER_PROFILE_FULL,
)
CONTROLLER_PROFILE_LABELS = {
    CONTROLLER_PROFILE_IDEAL: "ideal",
    CONTROLLER_PROFILE_ACTUATOR: "noisy",
    CONTROLLER_PROFILE_FULL: "full-sim",
}
DEFAULT_CONTROLLER_PROFILE_SETTINGS = {
    CONTROLLER_PROFILE_IDEAL: {
        "gimbal_kp": 3.0,
        "gimbal_kd": 2.765135066584,
    },
    CONTROLLER_PROFILE_ACTUATOR: {
        "gimbal_kp": 72.0,
        "gimbal_kd": 26.504134508781,
    },
    CONTROLLER_PROFILE_FULL: {
        "gimbal_kp": 22.25,
        "gimbal_kd": 3.124291252531,
    },
}
CONTROLLER_SETTING_LIMITS = {
    "gimbal_kp": (0.0, 100.0),
    "gimbal_kd": (0.0, 100.0),
}
AERO_SETTING_LIMITS = {
    "air_density_kg_m3": (0.0, 5.0),
    "drag_coefficient": (0.0, 3.0),
    "body_diameter_m": (0.001, 1.0),
    "normal_force_slope": (0.0, 50.0),
    "cp_cg_offset_m": (-5.0, 5.0),
    "pitch_damping_coefficient": (0.0, 10.0),
    "wind_x_m_s": (-100.0, 100.0),
}
ROCKET_SETTING_LIMITS = {
    "mass": (0.001, 100.0),
    "moment_arm": (0.0, 10.0),
    "pitch_moment_inertia": (0.0001, 100.0),
}
FIXED_Y_DOMAINS = {"gimbal_deg"}
SYMMETRIC_Y_DOMAINS = {
    "pitch_deg",
    "pitch_error_deg",
    "position_error_m",
    "angle_of_attack_deg",
    "normal_force_n",
    "aero_pitch_moment_n_m",
}
CHART_Y_DOMAINS = {
    "position_m": (-500, 2500),
    "x_m": (-250, 250),
    "y_m": (0, 2500),
    "pitch_deg": (-20, 20),
    "position_error_m": (-5, 5),
    "pitch_rate_deg_s": (-50, 50),
    "velocity_m_s": (-650, 350),
    "accel_m_s2": (-50, 50),
    "imu_accel_m_s2": (-5, 5),
    "gyro_deg_s": (-5, 5),
    "pitch_error_deg": (-0.5, 0.5),
    "pitch_rate_error_deg_s": (-10, 10),
    "accel_error_m_s2": (-2, 2),
    "gimbal_deg": GIMBAL_DOMAIN_IDEAL,
    "thrust_n": (0, 50),
    "dynamic_pressure_pa": (0, 10000),
    "drag_n": (0, 20),
    "angle_of_attack_deg": (-20, 20),
    "normal_force_n": (-20, 20),
    "aero_pitch_moment_n_m": (-2, 2),
}


def chart_y_domain(frame, columns, y_label):
    if y_label == "gimbal_deg" and frame.attrs.get("actuator_enabled"):
        return GIMBAL_DOMAIN_ACTUATOR

    default_domain = CHART_Y_DOMAINS.get(y_label)
    if default_domain is None:
        return None
    if y_label in FIXED_Y_DOMAINS:
        return default_domain

    values = frame[list(columns)].stack().dropna()
    if values.empty:
        return default_domain

    data_min = float(values.min())
    data_max = float(values.max())

    if y_label in SYMMETRIC_Y_DOMAINS:
        data_abs = max(abs(data_min), abs(data_max))
        if data_abs == 0:
            dmin, dmax = default_domain or (-1.0, 1.0)
            data_abs = max(abs(dmin), abs(dmax), 1.0)
        # Always ensure the scale max is slightly greater than the max |data point|
        domain_abs = data_abs * (1 + CHART_DOMAIN_PADDING)
        return (-domain_abs, domain_abs)

    # Non-symmetric: base the domain on the actual data range + padding so the
    # y-scale upper bound is always slightly above the highest data value.
    data_range = data_max - data_min
    if data_range <= 0:
        # Flat data: synthesize a small visible window
        if default_domain:
            def_min, def_max = default_domain
            def_span = def_max - def_min
            if def_span > 0:
                half = def_span * 0.1
                center = data_min
                return (center - half, center + half)
        pad = abs(data_max) * CHART_DOMAIN_PADDING if abs(data_max) > 1e-9 else CHART_DOMAIN_PADDING
        return (data_min - pad, data_max + pad)

    padding = data_range * CHART_DOMAIN_PADDING
    # Guarantee at least a tiny headroom even for low-variance signals
    min_pad = max(1e-9, abs(data_max) * 0.01)
    padding = max(padding, min_pad)

    dmin = data_min - padding
    dmax = data_max + padding

    # For non-negative signals (thrust, y_m, drag, etc.) keep lower bound at 0
    # when the data itself is non-negative. Upper bound still gets the +padding.
    if default_domain is not None:
        def_min, _ = default_domain
        if def_min == 0 and dmin < 0 <= data_min:
            dmin = 0

    return (dmin, dmax)


def burnout_time(frame):
    if "thrust_n" not in frame:
        return None

    thrusting_frame = frame[frame["thrust_n"] > BURNOUT_THRUST_THRESHOLD]
    if thrusting_frame.empty:
        return None

    return float(thrusting_frame["time_s"].iloc[-1])


def thrust_to_weight_summary(frame, mass):
    if frame.empty or "thrust_n" not in frame or mass <= 0.0:
        return None

    weight = mass * 9.81
    thrusting_frame = frame[frame["thrust_n"] > BURNOUT_THRUST_THRESHOLD]
    average_thrust = (
        float(thrusting_frame["thrust_n"].mean())
        if not thrusting_frame.empty
        else 0.0
    )
    max_thrust = float(frame["thrust_n"].max())
    return {
        "average": average_thrust / weight,
        "max": max_thrust / weight,
    }


def render_motor_sanity(config, frame):
    summary = thrust_to_weight_summary(frame, config.mass)
    if summary is None:
        return

    if summary["max"] < 1.2:
        st.warning(
            "Selected motor is below a practical liftoff margin: "
            f"max T/W {summary['max']:.2f}, average T/W "
            f"{summary['average']:.2f}."
        )
    elif summary["average"] < 1.0:
        st.warning(
            "Low apogee is expected with this motor and mass: "
            f"average T/W {summary['average']:.2f}, max T/W "
            f"{summary['max']:.2f}."
        )


def build_history_frame(rocket):
    rows = []
    for state in rocket.history:
        rows.append(
            {
                "time_s": state.time,
                "x_m": state.x,
                "y_m": state.y,
                "pitch_deg": math.degrees(state.pitch),
                "x_vel_m_s": state.x_vel,
                "y_vel_m_s": state.y_vel,
                "pitch_rate_deg_s": math.degrees(state.pitch_vel),
                "x_accel_m_s2": state.x_accel,
                "y_accel_m_s2": state.y_accel,
                "pitch_accel_deg_s2": math.degrees(state.pitch_accel),
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["actuator_enabled"] = rocket.actuator.config.enabled
    frame["thrust_n"] = [
        inputs.thrust for inputs in rocket.input_history[: len(frame)]
    ]
    frame["gimbal_deg"] = [
        math.degrees(inputs.gimbal_angle)
        for inputs in rocket.input_history[: len(frame)]
    ]
    frame["commanded_gimbal_deg"] = [
        math.degrees(
            inputs.commanded_gimbal_angle
            if inputs.commanded_gimbal_angle is not None
            else inputs.gimbal_angle
        )
        for inputs in rocket.input_history[: len(frame)]
    ]

    aero_results = getattr(rocket, "aero_history", [])[: len(frame)]

    def aero_column(get_value):
        values = [get_value(aero) for aero in aero_results]
        return values + [None] * (len(frame) - len(values))

    frame["dynamic_pressure_pa"] = aero_column(
        lambda aero: aero.dynamic_pressure_pa
    )
    frame["drag_n"] = aero_column(lambda aero: aero.drag_force_n)
    frame["angle_of_attack_deg"] = aero_column(
        lambda aero: math.degrees(aero.angle_of_attack_rad)
    )
    frame["normal_force_n"] = aero_column(lambda aero: aero.normal_force_n)
    frame["aero_pitch_moment_n_m"] = aero_column(
        lambda aero: aero.pitch_moment_n_m
    )
    frame["wind_x_m_s"] = aero_column(lambda aero: aero.wind_x_m_s)

    sensor_readings = rocket.sensor_history[: len(frame)]

    def sensor_column(get_value):
        values = [get_value(reading) for reading in sensor_readings]
        return values + [None] * (len(frame) - len(values))

    frame["sensor_pitch_rate_deg_s"] = sensor_column(
        lambda reading: math.degrees(reading.pitch_vel)
    )
    frame["sensor_x_accel_m_s2"] = sensor_column(
        lambda reading: reading.x_accel
    )
    frame["sensor_y_accel_m_s2"] = sensor_column(
        lambda reading: reading.y_accel
    )
    frame["sensor_pitch_rate_error_deg_s"] = (
        frame["sensor_pitch_rate_deg_s"] - frame["pitch_rate_deg_s"]
    )
    frame["sensor_x_accel_error_m_s2"] = (
        frame["sensor_x_accel_m_s2"] - frame["x_accel_m_s2"]
    )
    frame["sensor_y_accel_error_m_s2"] = (
        frame["sensor_y_accel_m_s2"] - frame["y_accel_m_s2"]
    )

    estimates = rocket.estimate_history[: len(frame)]

    def estimate_column(get_value):
        values = [get_value(estimate) for estimate in estimates]
        return values + [None] * (len(frame) - len(values))

    frame["estimated_x_m"] = estimate_column(lambda estimate: estimate.x)
    frame["estimated_y_m"] = estimate_column(lambda estimate: estimate.y)
    frame["estimated_x_vel_m_s"] = estimate_column(
        lambda estimate: estimate.x_vel
    )
    frame["estimated_y_vel_m_s"] = estimate_column(
        lambda estimate: estimate.y_vel
    )
    frame["estimated_pitch_deg"] = estimate_column(
        lambda estimate: math.degrees(estimate.pitch)
    )
    frame["estimated_pitch_rate_deg_s"] = estimate_column(
        lambda estimate: math.degrees(estimate.pitch_vel)
    )
    frame["estimated_pitch_error_deg"] = (
        frame["estimated_pitch_deg"] - frame["pitch_deg"]
    )
    frame["estimated_pitch_rate_error_deg_s"] = (
        frame["estimated_pitch_rate_deg_s"] - frame["pitch_rate_deg_s"]
    )
    frame["estimated_x_error_m"] = frame["estimated_x_m"] - frame["x_m"]
    frame["estimated_y_error_m"] = frame["estimated_y_m"] - frame["y_m"]
    return frame


def read_uploaded_curve(uploaded_curve):
    if uploaded_curve is None:
        return None

    return uploaded_curve.getvalue().decode("utf-8")


def controller_profile(
    actuator_enabled,
    sensor_noise_enabled=False,
    aero_enabled=False,
):
    if actuator_enabled and sensor_noise_enabled and aero_enabled:
        return CONTROLLER_PROFILE_FULL
    if actuator_enabled:
        return CONTROLLER_PROFILE_ACTUATOR
    return CONTROLLER_PROFILE_IDEAL


def controller_profile_for_mode(sim_mode):
    if sim_mode == SIM_MODE_FULL:
        return CONTROLLER_PROFILE_FULL
    if sim_mode == SIM_MODE_NOISY:
        return CONTROLLER_PROFILE_ACTUATOR
    return CONTROLLER_PROFILE_IDEAL


def mode_effects(sim_mode):
    return {
        "gimbal_enabled": True,
        "actuator_enabled": sim_mode in (SIM_MODE_NOISY, SIM_MODE_FULL),
        "sensor_noise_enabled": sim_mode in (SIM_MODE_NOISY, SIM_MODE_FULL),
        "aero_enabled": sim_mode == SIM_MODE_FULL,
    }


def validate_controller_settings(data):
    settings = {}
    if not isinstance(data, dict):
        return settings

    for key, (low, high) in CONTROLLER_SETTING_LIMITS.items():
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not low <= value <= high:
                continue
            settings[key] = float(value)
    return settings


def load_controller_settings():
    try:
        with CONTROLLER_SETTINGS_PATH.open(encoding="utf-8") as settings_file:
            data = json.load(settings_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    settings = {
        profile_name: validate_controller_settings(data.get(profile_name))
        for profile_name in CONTROLLER_PROFILES
    }

    legacy_settings = validate_controller_settings(data)
    if legacy_settings:
        settings[CONTROLLER_PROFILE_IDEAL].update(legacy_settings)
    return settings


def save_controller_settings(profile_name, gimbal_kp, gimbal_kd):
    CONTROLLER_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CONTROLLER_SETTINGS_PATH.with_suffix(".tmp")
    settings = load_controller_settings()
    settings[profile_name] = {
        "gimbal_kp": gimbal_kp,
        "gimbal_kd": gimbal_kd,
    }
    with temp_path.open("w", encoding="utf-8") as settings_file:
        json.dump(settings, settings_file, indent=2)
        settings_file.write("\n")
    temp_path.replace(CONTROLLER_SETTINGS_PATH)


def validate_aero_settings(data):
    settings = {}
    if not isinstance(data, dict):
        return settings

    aero_enabled = data.get("aero_enabled")
    if isinstance(aero_enabled, bool):
        settings["aero_enabled"] = aero_enabled

    for key, (low, high) in AERO_SETTING_LIMITS.items():
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not low <= value <= high:
                continue
            settings[key] = float(value)
    return settings


def read_aero_settings_data():
    try:
        with AERO_SETTINGS_PATH.open(encoding="utf-8") as settings_file:
            data = json.load(settings_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def load_aero_settings():
    data = read_aero_settings_data()
    settings = validate_aero_settings(data.get(AERO_SETTINGS_KEY))
    if settings:
        return settings
    return validate_aero_settings(data)


def save_aero_settings(
    aero_enabled,
    air_density_kg_m3,
    drag_coefficient,
    body_diameter_m,
    normal_force_slope,
    cp_cg_offset_m,
    pitch_damping_coefficient,
    wind_x_m_s,
):
    AERO_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = read_aero_settings_data()
    temp_path = AERO_SETTINGS_PATH.with_suffix(".tmp")
    data[AERO_SETTINGS_KEY] = {
        "aero_enabled": aero_enabled,
        "air_density_kg_m3": air_density_kg_m3,
        "drag_coefficient": drag_coefficient,
        "body_diameter_m": body_diameter_m,
        "normal_force_slope": normal_force_slope,
        "cp_cg_offset_m": cp_cg_offset_m,
        "pitch_damping_coefficient": pitch_damping_coefficient,
        "wind_x_m_s": wind_x_m_s,
    }
    with temp_path.open("w", encoding="utf-8") as settings_file:
        json.dump(data, settings_file, indent=2)
        settings_file.write("\n")
    temp_path.replace(AERO_SETTINGS_PATH)


def validate_rocket_settings(data):
    settings = {}
    if not isinstance(data, dict):
        return settings

    for key, (low, high) in ROCKET_SETTING_LIMITS.items():
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not low <= value <= high:
                continue
            settings[key] = float(value)
    return settings


def read_rocket_settings_data():
    try:
        with ROCKET_SETTINGS_PATH.open(encoding="utf-8") as settings_file:
            data = json.load(settings_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def load_rocket_settings():
    data = read_rocket_settings_data()
    settings = validate_rocket_settings(data.get(ROCKET_SETTINGS_KEY))
    if settings:
        return settings
    return validate_rocket_settings(data)


def save_rocket_settings(mass, moment_arm, pitch_moment_inertia):
    ROCKET_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = read_rocket_settings_data()
    temp_path = ROCKET_SETTINGS_PATH.with_suffix(".tmp")
    data[ROCKET_SETTINGS_KEY] = {
        "mass": mass,
        "moment_arm": moment_arm,
        "pitch_moment_inertia": pitch_moment_inertia,
    }
    with temp_path.open("w", encoding="utf-8") as settings_file:
        json.dump(data, settings_file, indent=2)
        settings_file.write("\n")
    temp_path.replace(ROCKET_SETTINGS_PATH)


def config_to_key(config):
    default_config = SimulationConfig()
    return tuple(
        getattr(config, field_name, getattr(default_config, field_name))
        for field_name in CONFIG_FIELDS
    )


def config_from_key(config_key):
    config_values = dict(zip(CONFIG_FIELDS, config_key))
    valid_fields = SimulationConfig.__dataclass_fields__
    return SimulationConfig(
        **{
            field_name: value
            for field_name, value in config_values.items()
            if field_name in valid_fields
        }
    )


@st.cache_data(show_spinner=False)
def run_simulation_frame(config_key, thrust_curve_text, history_frame_version):
    rocket = run_simulation(config_from_key(config_key), thrust_curve_text)
    return build_history_frame(rocket)


def downsample_frame(frame, max_points=MAX_CHART_POINTS):
    if len(frame) <= max_points:
        return frame

    stride = math.ceil(len(frame) / max_points)
    sampled = frame.iloc[::stride]
    if sampled.index[-1] == frame.index[-1]:
        return sampled

    return pd.concat([sampled, frame.iloc[[-1]]])


def sidebar_config():
    default_config = SimulationConfig()
    controller_settings = load_controller_settings()
    aero_settings = load_aero_settings()
    rocket_settings = load_rocket_settings()

    air_density_kg_m3 = aero_settings.get(
        "air_density_kg_m3",
        default_config.air_density_kg_m3,
    )
    drag_coefficient = aero_settings.get(
        "drag_coefficient",
        default_config.drag_coefficient,
    )
    body_diameter_m = aero_settings.get(
        "body_diameter_m",
        default_config.body_diameter_m,
    )
    cp_cg_offset_m = aero_settings.get(
        "cp_cg_offset_m",
        default_config.cp_cg_offset_m,
    )
    normal_force_slope = aero_settings.get(
        "normal_force_slope",
        default_config.normal_force_slope,
    )
    pitch_damping_coefficient = aero_settings.get(
        "pitch_damping_coefficient",
        default_config.pitch_damping_coefficient,
    )
    wind_x_m_s = aero_settings.get(
        "wind_x_m_s",
        default_config.wind_x_m_s,
    )
    mass = rocket_settings.get("mass", default_config.mass)
    moment_arm = rocket_settings.get("moment_arm", default_config.moment_arm)
    pitch_moment_inertia = rocket_settings.get(
        "pitch_moment_inertia",
        default_config.pitch_moment_inertia,
    )

    with st.sidebar:
        st.header("Inputs")
        selected_mode_label = st.radio(
            "Mode",
            [SIM_MODE_LABELS[mode] for mode in SIM_MODES],
            index=SIM_MODES.index(DEFAULT_SIM_MODE),
            horizontal=True,
            key="simulation_mode",
        )
        sim_mode = SIM_MODE_BY_LABEL[selected_mode_label]
        effects = mode_effects(sim_mode)
        gimbal_enabled = effects["gimbal_enabled"]
        actuator_enabled = effects["actuator_enabled"]
        sensor_noise_enabled = effects["sensor_noise_enabled"]
        aero_enabled = effects["aero_enabled"]

        st.subheader("Attitude")
        initial_pitch_deg = st.number_input(
            "Initial pitch (deg)",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=1.0,
        )

        st.subheader("Controller")
        gimbal_enabled = st.toggle(
            "Gimbal control",
            value=effects["gimbal_enabled"],
            key="gimbal_enabled",
        )
        active_controller_profile = controller_profile_for_mode(sim_mode)
        active_controller_settings = controller_settings.get(
            active_controller_profile,
            {},
        )
        active_controller_defaults = DEFAULT_CONTROLLER_PROFILE_SETTINGS[
            active_controller_profile
        ]
        active_controller_label = CONTROLLER_PROFILE_LABELS[
            active_controller_profile
        ]
        gimbal_kp = st.number_input(
            f"Gimbal kP ({active_controller_label})",
            min_value=0.0,
            max_value=100.0,
            value=active_controller_settings.get(
                "gimbal_kp",
                active_controller_defaults["gimbal_kp"],
            ),
            step=0.01,
            format="%.4f",
            disabled=not gimbal_enabled,
            key=f"gimbal_kp_{active_controller_profile}",
        )
        gimbal_kd = st.number_input(
            f"Gimbal kD ({active_controller_label})",
            min_value=0.0,
            max_value=100.0,
            value=active_controller_settings.get(
                "gimbal_kd",
                active_controller_defaults["gimbal_kd"],
            ),
            step=0.001,
            format="%.4f",
            disabled=not gimbal_enabled,
            key=f"gimbal_kd_{active_controller_profile}",
        )
        if st.button(f"Save {active_controller_label} kP/kD"):
            try:
                save_controller_settings(
                    active_controller_profile,
                    gimbal_kp,
                    gimbal_kd,
                )
            except OSError as exc:
                st.error(
                    f"Could not save {active_controller_label} kP/kD: {exc}"
                )
            else:
                st.success(f"Saved {active_controller_label} kP/kD")

        if sim_mode == SIM_MODE_FULL:
            st.subheader("Aerodynamics")
            air_density_kg_m3 = st.number_input(
                "Air density (kg/m^3)",
                min_value=0.0,
                max_value=5.0,
                value=air_density_kg_m3,
                step=0.025,
                format="%.3f",
                disabled=not aero_enabled,
            )
            drag_coefficient = st.number_input(
                "Drag coefficient",
                min_value=0.0,
                max_value=3.0,
                value=drag_coefficient,
                step=0.05,
                format="%.3f",
                disabled=not aero_enabled,
            )
            body_diameter_m = st.number_input(
                "Body diameter (m)",
                min_value=0.001,
                max_value=1.0,
                value=body_diameter_m,
                step=0.005,
                format="%.3f",
                disabled=not aero_enabled,
            )
            cp_cg_offset_m = st.number_input(
                "CP-CG offset (m)",
                min_value=-5.0,
                max_value=5.0,
                value=cp_cg_offset_m,
                step=0.01,
                format="%.3f",
                disabled=not aero_enabled,
            )
            normal_force_slope = st.number_input(
                "Normal force slope",
                min_value=0.0,
                max_value=50.0,
                value=normal_force_slope,
                step=0.1,
                format="%.3f",
                disabled=not aero_enabled,
            )
            pitch_damping_coefficient = st.number_input(
                "Pitch damping",
                min_value=0.0,
                max_value=10.0,
                value=pitch_damping_coefficient,
                step=0.005,
                format="%.4f",
                disabled=not aero_enabled,
            )
            wind_x_m_s = st.number_input(
                "Horizontal wind (m/s)",
                min_value=-100.0,
                max_value=100.0,
                value=wind_x_m_s,
                step=0.5,
                format="%.2f",
                disabled=not aero_enabled,
            )
            if st.button("Save aerodynamics"):
                try:
                    save_aero_settings(
                        aero_enabled,
                        air_density_kg_m3,
                        drag_coefficient,
                        body_diameter_m,
                        normal_force_slope,
                        cp_cg_offset_m,
                        pitch_damping_coefficient,
                        wind_x_m_s,
                    )
                except OSError as exc:
                    st.error(f"Could not save aerodynamics: {exc}")
                else:
                    st.success("Saved aerodynamics")

        st.subheader("Rocket")
        mass = st.number_input(
            "Mass (kg)",
            min_value=0.001,
            max_value=100.0,
            value=mass,
            step=0.05,
            key="rocket_mass_v2",
        )
        moment_arm = st.number_input(
            "Moment arm (m)",
            min_value=0.0,
            max_value=10.0,
            value=moment_arm,
            step=0.05,
            key="rocket_moment_arm_v2",
        )
        pitch_moment_inertia = st.number_input(
            "Pitch MMOI (kg m^2)",
            min_value=0.0001,
            max_value=100.0,
            value=pitch_moment_inertia,
            step=0.01,
            format="%.4f",
            key="rocket_pitch_mmoi_v2",
        )
        if st.button("Save rocket characteristics"):
            try:
                save_rocket_settings(
                    mass,
                    moment_arm,
                    pitch_moment_inertia,
                )
            except OSError as exc:
                st.error(f"Could not save rocket characteristics: {exc}")
            else:
                st.success("Saved rocket characteristics")

        with st.expander("Advanced"):
            max_time = st.number_input(
                "Max time (s)",
                min_value=0.1,
                max_value=300.0,
                value=60.0,
                step=1.0,
            )
            time_step = st.number_input(
                "Time step (s)",
                min_value=0.001,
                max_value=0.1,
                value=0.005,
                step=0.001,
                format="%.3f",
            )
            uploaded_curve = st.file_uploader("Thrust curve CSV", type=["csv"])

    config = SimulationConfig(
        time_step=time_step,
        max_time=max_time,
        mass=mass,
        moment_arm=moment_arm,
        pitch_moment_inertia=pitch_moment_inertia,
        initial_pitch_deg=initial_pitch_deg,
        gimbal_enabled=gimbal_enabled,
        gimbal_kp=gimbal_kp,
        gimbal_kd=gimbal_kd,
        actuator_enabled=actuator_enabled,
        actuator_max_angle_deg=default_config.actuator_max_angle_deg,
        actuator_slew_rate_deg_s=default_config.actuator_slew_rate_deg_s,
        actuator_delay_s=default_config.actuator_delay_s,
        actuator_lag_s=default_config.actuator_lag_s,
        actuator_deadband_deg=default_config.actuator_deadband_deg,
        actuator_bias_deg=default_config.actuator_bias_deg,
        sensor_noise_enabled=sensor_noise_enabled,
        sensor_random_seed=default_config.sensor_random_seed,
        gyro_noise_deg_s=default_config.gyro_noise_deg_s,
        accel_noise_m_s2=default_config.accel_noise_m_s2,
        aero_enabled=aero_enabled,
        air_density_kg_m3=air_density_kg_m3,
        drag_coefficient=drag_coefficient,
        body_diameter_m=body_diameter_m,
        normal_force_slope=normal_force_slope,
        cp_cg_offset_m=cp_cg_offset_m,
        pitch_damping_coefficient=pitch_damping_coefficient,
        wind_x_m_s=wind_x_m_s,
    )
    return config, uploaded_curve


def render_metrics(frame):
    max_altitude = frame["y_m"].max()
    max_range = frame["x_m"].abs().max()
    airborne_frame = frame[frame["y_m"] > 0]
    max_speed_frame = airborne_frame if not airborne_frame.empty else frame.iloc[:1]
    max_speed = (
        max_speed_frame["x_vel_m_s"] ** 2
        + max_speed_frame["y_vel_m_s"] ** 2
    ) ** 0.5
    final_pitch = frame["pitch_deg"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Apogee", f"{max_altitude:.1f} m")
    col2.metric("Max range", f"{max_range:.1f} m")
    col3.metric("Max speed", f"{max_speed.max():.1f} m/s")
    col4.metric("Final pitch", f"{final_pitch:.1f} deg")


def render_line_chart(
    chart_frame,
    export_frame,
    columns,
    x_column,
    y_label,
):
    chart_data = chart_frame[[x_column, *columns]].melt(
        id_vars=x_column,
        var_name="series",
        value_name=CHART_VALUE_COLUMN,
    )
    y_domain = chart_y_domain(export_frame, columns, y_label)
    y_scale = alt.Scale(
        domain=y_domain,
        nice=False if y_label == "gimbal_deg" else True,
    )
    chart = alt.Chart(chart_data).mark_line().encode(
        x=alt.X(
            f"{x_column}:Q",
            title=x_column,
            scale=alt.Scale(domain=[0, export_frame[x_column].max()]),
        ),
        y=alt.Y(
            f"{CHART_VALUE_COLUMN}:Q",
            title=y_label,
            scale=y_scale,
            axis=alt.Axis(
                labelPadding=4,
                minExtent=42,
                titlePadding=6,
            ),
        ),
        tooltip=[
            alt.Tooltip(f"{x_column}:Q", title=x_column),
            alt.Tooltip("series:N", title="series"),
            alt.Tooltip(f"{CHART_VALUE_COLUMN}:Q", title=y_label),
        ],
    )
    if len(columns) > 1:
        chart = chart.encode(
            color=alt.Color(
                "series:N",
                title=None,
                legend=alt.Legend(
                    orient="top",
                    direction="horizontal",
                ),
            )
        )

    burnout = burnout_time(export_frame)
    if x_column == "time_s" and burnout is not None:
        burnout_data = pd.DataFrame(
            {"time_s": [burnout], "event": ["burnout"]}
        )
        burnout_rule = alt.Chart(burnout_data).mark_rule(
            color="#ef4444",
            strokeWidth=2,
            strokeDash=[6, 4],
        ).encode(
            x=alt.X("time_s:Q"),
            tooltip=[
                alt.Tooltip("event:N", title="event"),
                alt.Tooltip("time_s:Q", title="time_s", format=".3f"),
            ],
        )
        chart = alt.layer(chart, burnout_rule)

    zoom = alt.selection_interval(
        bind="scales",
        encodings=["x", "y"],
        zoom="wheel!",
    )
    chart = chart.add_params(zoom)
    chart = chart.properties(padding={"left": 4})

    st.altair_chart(chart, width="stretch")


def render_graphs_view(chart_frame, export_frame):
    col1, col2 = st.columns(2)
    with col1:
        render_line_chart(
            chart_frame,
            export_frame,
            ["y_m", "estimated_y_m"],
            "time_s",
            "y_m",
        )
    with col2:
        render_line_chart(
            chart_frame,
            export_frame,
            ["x_m", "estimated_x_m"],
            "time_s",
            "x_m",
        )

    col1, col2 = st.columns(2)
    with col1:
        render_line_chart(
            chart_frame,
            export_frame,
            ["pitch_deg", "estimated_pitch_deg"],
            "time_s",
            "pitch_deg",
        )
    with col2:
        render_line_chart(
            chart_frame,
            export_frame,
            ["commanded_gimbal_deg", "gimbal_deg"],
            "time_s",
            "gimbal_deg",
        )

    col1, col2 = st.columns(2)
    with col1:
        render_line_chart(
            chart_frame,
            export_frame,
            ["estimated_pitch_error_deg"],
            "time_s",
            "pitch_error_deg",
        )
    with col2:
        render_line_chart(
            chart_frame,
            export_frame,
            ["estimated_x_error_m", "estimated_y_error_m"],
            "time_s",
            "position_error_m",
        )


def render_sensor_graphs_view(chart_frame, export_frame):
    col1, col2 = st.columns(2)
    with col1:
        render_line_chart(
            chart_frame,
            export_frame,
            ["sensor_x_accel_m_s2", "sensor_y_accel_m_s2"],
            "time_s",
            "imu_accel_m_s2",
        )
    with col2:
        render_line_chart(
            chart_frame,
            export_frame,
            ["sensor_pitch_rate_deg_s"],
            "time_s",
            "gyro_deg_s",
        )


def render_other_graphs_view(chart_frame, export_frame):
    col1, col2 = st.columns(2)
    with col1:
        render_line_chart(
            chart_frame,
            export_frame,
            ["x_vel_m_s", "y_vel_m_s"],
            "time_s",
            "velocity_m_s",
        )
    with col2:
        render_line_chart(
            chart_frame,
            export_frame,
            ["thrust_n"],
            "time_s",
            "thrust_n",
        )


def render_aero_graphs_view(chart_frame, export_frame):
    col1, col2 = st.columns(2)
    with col1:
        render_line_chart(
            chart_frame,
            export_frame,
            ["dynamic_pressure_pa"],
            "time_s",
            "dynamic_pressure_pa",
        )
    with col2:
        render_line_chart(
            chart_frame,
            export_frame,
            ["drag_n"],
            "time_s",
            "drag_n",
        )

    col1, col2 = st.columns(2)
    with col1:
        render_line_chart(
            chart_frame,
            export_frame,
            ["angle_of_attack_deg"],
            "time_s",
            "angle_of_attack_deg",
        )
    with col2:
        render_line_chart(
            chart_frame,
            export_frame,
            ["aero_pitch_moment_n_m"],
            "time_s",
            "aero_pitch_moment_n_m",
        )


def render_data_view(frame):
    rows_to_show = st.slider(
        "Rows to preview",
        min_value=100,
        max_value=len(frame),
        value=min(DEFAULT_DATA_ROWS, len(frame)),
        step=100,
    )
    st.dataframe(
        frame.head(rows_to_show),
        width="stretch",
        hide_index=True,
        height=460,
    )


def render_charts(frame):
    chart_frame = downsample_frame(frame)
    view = st.radio(
        "View",
        ["Graphs", "Sensors", "Aero", "Other", "Data"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view == "Graphs":
        render_graphs_view(chart_frame, frame)
    elif view == "Sensors":
        render_sensor_graphs_view(chart_frame, frame)
    elif view == "Aero":
        render_aero_graphs_view(chart_frame, frame)
    elif view == "Other":
        render_other_graphs_view(chart_frame, frame)
    else:
        render_data_view(frame)


def main():
    st.set_page_config(page_title="TVC Sim", layout="wide")
    st.title("TVC Sim")

    config, uploaded_curve = sidebar_config()
    thrust_curve_text = read_uploaded_curve(uploaded_curve)

    try:
        frame = run_simulation_frame(
            config_to_key(config),
            thrust_curve_text,
            HISTORY_FRAME_VERSION,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        st.error(str(exc))
        return

    render_motor_sanity(config, frame)
    render_metrics(frame)
    render_charts(frame)


if __name__ == "__main__":
    main()
