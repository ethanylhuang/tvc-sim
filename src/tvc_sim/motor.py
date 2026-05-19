import csv
import io
from importlib import resources
from pathlib import Path

import numpy as np


DEFAULT_THRUST_CURVE = "AeroTech_H13ST.csv"


def default_thrust_curve_path():
    return resources.files("tvc_sim").joinpath("data", DEFAULT_THRUST_CURVE)


def load_thrust_curve(filename, max_time, time_step):
    path = Path(filename)
    if not path.exists() and path.name == DEFAULT_THRUST_CURVE:
        path = default_thrust_curve_path()

    with open(path, "r") as f:
        data_string = f.read()

    return load_thrust_curve_from_text(data_string, max_time, time_step)


def load_thrust_curve_from_text(data_string, max_time, time_step):
    reader = csv.reader(io.StringIO(data_string.strip()))

    raw_times = []
    raw_thrusts = []

    for row in reader:
        try:
            raw_times.append(float(row[0]))
            raw_thrusts.append(float(row[1]))
        except (IndexError, ValueError):
            continue

    if not raw_times:
        raise ValueError("No thrust curve samples found.")

    times = np.arange(0, max_time, time_step)
    thrusts = np.interp(times, raw_times, raw_thrusts, left=0, right=0)

    return times, thrusts
