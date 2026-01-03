import numpy as np
from scipy import interpolate
import csv
import io
import matplotlib.pyplot as plt
from rocket import Rocket

THRUST_CURVE_FILENAME = "AeroTech_H13ST.csv"
TIME_STEP = 0.005
MAX_TIME = 60
MOMENT_INERTIA = 0.06
MOMENT_ARM = 0.4
MASS = 0.75
INITIAL_ANGLE = 10


def main():
    rocket = Rocket(
        TIME_STEP,
        MAX_TIME,
        MASS,
        MOMENT_ARM,
        MOMENT_INERTIA,
        INITIAL_ANGLE,
        THRUST_CURVE_FILENAME,
    )
    rocket.run_sim()

    plt.figure(figsize=(12, 5), dpi=100)

    plt.subplot(1, 2, 1)
    plt.plot(rocket.times, rocket.y_arr)
    plt.xlabel("Time")
    plt.ylabel("Y (m)")
    plt.title("Altitude vs. Time")
    plt.ylim(0, 3000)

    plt.subplot(1, 2, 2)
    plt.plot(rocket.times, rocket.x_arr)
    plt.xlabel("Time")
    plt.ylabel("X (m)")
    plt.title("Horizontal Position vs. Time")
    plt.ylim(0, 3000)

    plt.suptitle("ROCKET SIMULATION TRAJECTORY")
    plt.show()


if __name__ == "__main__":
    main()
