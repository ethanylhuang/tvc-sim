import numpy as np
from scipy import interpolate
import csv
import io
import matplotlib.pyplot as plt


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
    ):
        self.time = 0
        self.max_time = max_time
        self.dt = time_step
        self.mass = mass
        self.moment_inertia = moment_inertia
        self.moment_arm = moment_arm
        self.times, self.thrusts = self.import_thrust_curve(thrust_curve_filename)

        self.x = 0
        self.y = 0
        self.theta = initial_angle
        self.gimbal_theta = 0

        self.x_vel = 0
        self.y_vel = 0
        self.theta_vel = 0

        self.x_accel = 0
        self.y_accel = 0
        self.theta_accel = 0

        self.x_arr = []
        self.y_arr = []
        self.theta_arr = []

    def import_thrust_curve(self, filename):
        with open(filename, "r") as f:
            data_string = f.read()

        reader = csv.reader(io.StringIO(data_string.strip()))

        raw_times = []
        raw_thrusts = []

        for row in reader:
            try:
                raw_times.append(float(row[0]))
                raw_thrusts.append(float(row[1]))
            except:
                continue

        times = np.arange(0, self.max_time, self.dt)
        thrusts = np.interp(times, raw_times, raw_thrusts, left=0, right=0)

        return times, thrusts

    def step(self):
        self.x_arr.append(self.x)
        self.y_arr.append(self.y)
        self.theta_arr.append(self.theta)

        index = int(self.time / self.dt)
        thrust = self.thrusts[index]
        self.dynamics(thrust)

        self.integrate()
        self.time += self.dt

    def dynamics(self, thrust):
        angle_rad = self.theta * (np.pi / 180)
        x_thrust = thrust * np.sin(angle_rad)
        y_thrust = thrust * np.cos(angle_rad)

        gimbal_rad = np.radians(self.gimbal_theta)
        torque = thrust * np.sin(gimbal_rad) * self.moment_arm

        y_net = y_thrust - (9.81 * self.mass)

        self.x_accel = x_thrust / self.mass
        self.y_accel = y_net / self.mass
        self.theta_accel = torque / self.moment_inertia

    def integrate(self):
        self.x_vel += self.x_accel * self.dt
        self.y_vel += self.y_accel * self.dt
        self.theta_vel += self.theta_accel * self.dt

        self.y += self.y_vel * self.dt
        if self.y <= 0:
            self.y = 0
            return

        self.x += self.x_vel * self.dt
        self.theta += self.theta_vel * self.dt

    def run_sim(self):
        self.time = 0
        while self.time < self.max_time:
            self.step()
