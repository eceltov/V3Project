import torch
import lib.processingTool as pt
import lib.rankCalculations as rc
import lib.boundaries as boundaries
import lib.rectangles as rectangles

def get_representing_rects(rects, kmeans_k, representing_rect_area):
  centroids, sorted_rects = rectangles.get_centroids_and_sorted_rects(rects, kmeans_k)
  representing_rects = []
  for i in range(kmeans_k):
    # draw a rectangle around the centroid
    representing_rect = rectangles.get_representing_rect(centroids[i], sorted_rects[i], representing_rect_area)
    # move the rectangle so that it is fully on the frame
    moved_representing_rect = rectangles.confine_to_area(pt.frame_width, pt.frame_height, representing_rect)
    representing_rects.append(moved_representing_rect)
  return representing_rects
