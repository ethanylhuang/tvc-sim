import matplotlib.pyplot as plt


def plot_trajectory(rocket):
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
