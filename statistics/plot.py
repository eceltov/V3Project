import matplotlib.pyplot as plt
import numpy as np

BAR_WIDTH = 0.7


def bar_chart(results: list[tuple[str, float]]) -> None:
    x_names = [result[0] for result in results]
    x_values = [result[1] for result in results]

    plt.bar(x_names, x_values, width=BAR_WIDTH)

    # Annotations
    plt.ylabel("Values")
    plt.title(f"Performance")

    # y-axis lines for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()


def cdf_chart(results: list[tuple[str, list[int]]], logarithmic: bool = False) -> None:
    # Plot the curve for each model
    for model_name, ranks in results:
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)

        plt.plot(sorted_ranks, cdf, label=model_name)

    # Logarithmic scale to enhance the difference in first ranks
    if logarithmic:
        plt.xscale("log")

    plt.title("Cumulative Distribution of Ranks")
    plt.xlabel("Rank")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    bar_chart([("A", 41), ("B", 84), ("C", 63)])
