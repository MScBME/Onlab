import matplotlib.pyplot as plt


def plot_speed(times, speeds):

    plt.figure()

    plt.plot(times, speeds)

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (px/s)")
    plt.title("Swimmer speed")

    plt.show()