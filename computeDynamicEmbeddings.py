import sys
import datetime

print(f"Job started at: {datetime.datetime.now()}")

import lib.rectangles as rectangles

# rects = [
#   [0, 0, 50, 100],
#   [0, 100, 50, 200],
#   [500, 0, 550, 100],
#   [500, 100, 550, 200],
# ]
# centroids, sorted_rects = rectangles.get_centroids_and_sorted_rects(rects, 2)
# print(centroids)
# print(sorted_rects)

# representing_rect = rectangles.get_representing_rect(centroids[0], sorted_rects[0], 5000)
# print(representing_rect)

rect = [
  -10,
  10,
  90,
  110,
]
print(rectangles.confine_to_area(100, 100, rect))
