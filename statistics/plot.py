import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    bar_chart([("A", 41), ("B", 84), ("C", 63)])
