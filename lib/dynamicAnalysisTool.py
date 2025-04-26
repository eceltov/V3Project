import torch
import lib.processingTool as pt
import lib.rankCalculations as rc
import lib.boundaries as boundaries
import lib.rectangles as rectangles

def get_representing_rects(rects: list[list], kmeans_k: int, representing_rect_area: int) -> list[list]:
  """Takes a list of rectangles and clusters them using the kmeans algorithm.
      Defines new rectangles centered around kmean clusters with a given area (all such rectangles have the same area).
      These rectangles have a shape derived from the shape of the input rectangles linked to the matching centroid
      (widths and heights are summed and divided by the same factor to match the @representing_rect_area)
      In case the representing rectangles are outside of the frame, they are shifted.

  Args:
      rects (list[list]): A list of [x1, y1, x2, y2] lists representing input rectangles.
      kmeans_k (int): How many clusters should there be.
      representing_rect_area (int): The area (in pixels) of how big each representing rectangle should be.

  Returns:
      list[list]: Returns a list of [x1, y1, x2, y2] lists representing the rectangles centered around kmeans centroids.
      This list is not sorted and can be ordered differently in identical calls.
  """

  # get centroids and linked rects
  centroids, sorted_rects = rectangles.get_centroids_and_sorted_rects(rects, kmeans_k)
  representing_rects = []
  for i in range(kmeans_k):
    # draw a rectangle around the centroid
    representing_rect = rectangles.get_representing_rect(centroids[i], sorted_rects[i], representing_rect_area)
    # move the rectangle so that it is fully in the frame
    moved_representing_rect = rectangles.confine_to_area(pt.frame_width, pt.frame_height, representing_rect)
    representing_rects.append(moved_representing_rect)
  return representing_rects
