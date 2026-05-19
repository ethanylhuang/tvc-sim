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
    "actuator_noise_deg",
    "actuator_random_seed",
    "sensor_noise_enabled",
    "sensor_random_seed",
    "gyro_noise_deg_s",
    "accel_noise_m_s2",
)
MAX_CHART_POINTS = 1200
DEFAULT_DATA_ROWS = 500
HISTORY_FRAME_VERSION = 13
CHART_VALUE_COLUMN = "__chart_value"
CHART_DOMAIN_PADDING = 0.05
BURNOUT_THRUST_THRESHOLD = 0.0
GIMBAL_DOMAIN_IDEAL = (-0.02, 0.02)
GIMBAL_DOMAIN_ACTUATOR = (-0.1, 0.1)
CONTROLLER_SETTINGS_PATH = (
    Path.home() / ".tvc_sim" / "controller_settings.json"
)
CONTROLLER_PROFILE_IDEAL = "ideal"
CONTROLLER_PROFILE_ACTUATOR = "actuator"
CONTROLLER_PROFILES = (
    CONTROLLER_PROFILE_IDEAL,
    CONTROLLER_PROFILE_ACTUATOR,
)
CONTROLLER_PROFILE_LABELS = {
    CONTROLLER_PROFILE_IDEAL: "ideal",
    CONTROLLER_PROFILE_ACTUATOR: "actuator-on",
}
CONTROLLER_SETTING_LIMITS = {
    "gimbal_kp": (0.0, 100.0),
    "gimbal_kd": (0.0, 100.0),
}
FIXED_Y_DOMAINS = {"gimbal_deg"}
SYMMETRIC_Y_DOMAINS = {
    "pitch_deg",
    "pitch_error_deg",
    "position_error_m",
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
    default_min, default_max = default_domain

    if y_label in SYMMETRIC_Y_DOMAINS:
        default_abs = max(abs(default_min), abs(default_max))
        data_abs = max(abs(data_min), abs(data_max))
        domain_abs = max(default_abs, data_abs)
        if domain_abs > default_abs:
            domain_abs *= 1 + CHART_DOMAIN_PADDING
        return (-domain_abs, domain_abs)

    domain_min = min(default_min, data_min)
    domain_max = max(default_max, data_max)
    if domain_min == default_min and domain_max == default_max:
        return default_domain

    padding = (domain_max - domain_min) * CHART_DOMAIN_PADDING
    if data_min < default_min:
        domain_min -= padding
    if data_max > default_max:
        domain_max += padding
    return (domain_min, domain_max)


def burnout_time(frame):
    if "thrust_n" not in frame:
        return None

    thrusting_frame = frame[frame["thrust_n"] > BURNOUT_THRUST_THRESHOLD]
    if thrusting_frame.empty:
        return None

    return float(thrusting_frame["time_s"].iloc[-1])


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


def controller_profile(actuator_enabled):
    if actuator_enabled:
        return CONTROLLER_PROFILE_ACTUATOR
    return CONTROLLER_PROFILE_IDEAL


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
    actuator_enabled_state = st.session_state.get(
        "actuator_enabled",
        default_config.actuator_enabled,
    )

    with st.sidebar:
        st.header("Inputs")
        st.subheader("Attitude")
        initial_pitch_deg = st.number_input(
            "Initial pitch (deg)",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=1.0,
        )

        st.subheader("Controller")
        gimbal_enabled = st.toggle("Gimbal control", value=True)
        active_controller_profile = controller_profile(actuator_enabled_state)
        active_controller_settings = controller_settings.get(
            active_controller_profile,
            {},
        )
        active_controller_label = CONTROLLER_PROFILE_LABELS[
            active_controller_profile
        ]
        gimbal_kp = st.number_input(
            f"Gimbal kP ({active_controller_label})",
            min_value=0.0,
            max_value=100.0,
            value=active_controller_settings.get(
                "gimbal_kp",
                default_config.gimbal_kp,
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
                default_config.gimbal_kd,
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

        st.subheader("Actuator")
        actuator_enabled = st.toggle(
            "Actuator imperfections",
            value=default_config.actuator_enabled,
            key="actuator_enabled",
        )

        st.subheader("Sensors")
        sensor_noise_enabled = st.toggle(
            "Sensor noise",
            value=default_config.sensor_noise_enabled,
            key="sensor_noise_enabled",
        )
        st.subheader("Rocket")
        mass = st.number_input(
            "Mass (kg)",
            min_value=0.001,
            max_value=100.0,
            value=0.75,
            step=0.05,
        )
        moment_arm = st.number_input(
            "Moment arm (m)",
            min_value=0.0,
            max_value=10.0,
            value=0.4,
            step=0.05,
        )
        pitch_moment_inertia = st.number_input(
            "Pitch MMOI (kg m^2)",
            min_value=0.0001,
            max_value=100.0,
            value=0.06,
            step=0.01,
            format="%.4f",
        )

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
        actuator_noise_deg=default_config.actuator_noise_deg,
        actuator_random_seed=default_config.actuator_random_seed,
        sensor_noise_enabled=sensor_noise_enabled,
        sensor_random_seed=default_config.sensor_random_seed,
        gyro_noise_deg_s=default_config.gyro_noise_deg_s,
        accel_noise_m_s2=default_config.accel_noise_m_s2,
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
        ["Graphs", "Sensors", "Other", "Data"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view == "Graphs":
        render_graphs_view(chart_frame, frame)
    elif view == "Sensors":
        render_sensor_graphs_view(chart_frame, frame)
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

    render_metrics(frame)
    render_charts(frame)


if __name__ == "__main__":
    main()
