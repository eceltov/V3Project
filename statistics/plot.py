import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

BAR_WIDTH = 0.7
IMAGE_WIDTH = 249
IMAGE_HEIGHT = 140


def bar_chart(results: list[tuple[str, float]], out_dest: str | None = None) -> None:
    x_names = [result[0] for result in results]
    x_values = [result[1] for result in results]

    plt.bar(x_names, x_values, width=BAR_WIDTH)

    # Annotations
    plt.ylabel("Average Rank")
    plt.title(f"Performance")

    # y-axis lines for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if out_dest is not None:
        plt.savefig(out_dest)
    else:
        plt.show()


def cdf_chart(results: list[tuple[str, list[int]]], logarithmic: bool = False, out_dest: str | None = None) -> None:
    # Plot the curve for each model
    for model_name, ranks in results:
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)

        plt.plot(sorted_ranks, cdf, label=model_name)

    # Logarithmic scale to enhance the difference in first ranks
    if logarithmic:
        plt.xscale("log")

    plt.title("Cumulative Distribution of Ranks")
    plt.xlabel(f'Rank{" (log scale)" if logarithmic else ""}')
    plt.ylabel("Fraction of Images ...")
    plt.legend()
    plt.grid(True)

    if out_dest is not None:
        plt.savefig(out_dest)
    else:
        plt.show()


def violin_chart(results: list[tuple[str, list[int]]], logarithmic: bool = False, out_dest: str | None = None) -> None:
    data = [model_results[1] for model_results in results]
    labels = [model_results[0] for model_results in results]

    plt.violinplot(data, showmeans=True, showmedians=True)

    # Logarithmic scale to enhance the difference in first ranks
    if logarithmic:
        plt.yscale("log")

    plt.title("Rank Distribution")
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.xlabel("Model")
    plt.ylabel(f'Rank{" (log scale)" if logarithmic else ""}')

    if out_dest is not None:
        plt.savefig(out_dest)
    else:
        plt.show()


def _size(rank: int) -> float:
    return (np.log(25000) - np.log(rank + 1)) / 2


def heat_map(object_locations: list[tuple[tuple[int, int], tuple[int, int]]], out_dest: str | None = None) -> None:
    # 2D grid to count overlaps
    overlap_grid = [[0] * IMAGE_WIDTH for _ in range(IMAGE_HEIGHT)]

    # Count overlaps
    for (start_x, start_y), (end_x, end_y) in object_locations:

        start_x, end_x = max(0, min(start_x, end_x)), min(IMAGE_WIDTH - 1, max(start_x, end_x))
        start_y, end_y = max(0, min(start_y, end_y)), min(IMAGE_HEIGHT - 1, max(start_y, end_y))

        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                overlap_grid[y][x] += 1

    max_overlap = max(max(row) for row in overlap_grid)

    img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            overlap_count = overlap_grid[y][x]

            if max_overlap > 0:
                brightness = 255 * (1 - overlap_count / max_overlap)
            else:
                # No overlap at all
                brightness = 255

            draw.point((x, y), fill=int(brightness))

    if out_dest is not None:
        img.save(out_dest)
    else:
        img.show()
