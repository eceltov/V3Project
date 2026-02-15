from sklearn.cluster import KMeans
import math
import random

# swaps rect coords so that the first point has all dimensions lower than the second
def normalize(rect):
  x1, y1, x2, y2 = rect
  if x1 > x2:
    x1, x2 = x2, x1
  if y1 > y2:
    y1, y2 = y2, y1
  return [x1, y1, x2, y2]

def normalize_recall_annotation(recall_annotation):
  annotation = recall_annotation["annotation"]
  for round in annotation["rounds"]:
    for rect_id in ["initialRect", "finalRect"]:
      rect = round[rect_id]
      x = math.floor(rect["x"])
      y = math.floor(rect["y"])
      width = math.floor(rect["width"])
      height = math.floor(rect["height"])
      rect = [x, y, x + width, y + height]
      round[rect_id] = rect

def get_IoU(rect1, rect2):
  # normalize coords
  rect1 = normalize(rect1)
  rect2 = normalize(rect2)

  # compute intersection area
  x_left = max(rect1[0], rect2[0])
  y_top = max(rect1[1], rect2[1])
  x_right = min(rect1[2], rect2[2])
  y_bottom = min(rect1[3], rect2[3])

  if x_right < x_left or y_bottom < y_top:
      return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute rectangle areas
  rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
  rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

  iou = intersection_area / float(rect1_area + rect2_area - intersection_area)
  return iou

# returns the index of the segment with the highest IoU with the source rect
def get_best_IoU_segment_idx(rect, segment_rects):
  best_IoU = get_IoU(rect, segment_rects[0])
  best_segment_idx = 0
  for segment_idx in range(1, len(segment_rects)):
    IoU = get_IoU(rect, segment_rects[segment_idx])
    if IoU > best_IoU:
      best_IoU = IoU
      best_segment_idx = segment_idx

  return best_segment_idx, best_IoU

def get_area(rect):
  x1, y1, x2, y2 = normalize(rect)
  return (x2 - x1) * (y2 - y1)

def get_centerpoint(rect):
  x1, y1, x2, y2 = rect
  x = (x1 + x2) // 2
  y = (y1 + y2) // 2
  return (x, y)

# returns kmeans centroids and lists of rects aligned to the centroids 
def get_centroids_and_sorted_rects(rects, k):
  centres = [get_centerpoint(rect) for rect in rects]
  kmeans = KMeans(n_clusters=k, n_init=10)
  kmeans.fit(centres)
  centroids = kmeans.cluster_centers_
  labels = kmeans.labels_

  sorted_rects = [[] for _ in range(k)]
  for i in range(len(labels)):
    label = labels[i]
    rect = rects[i]
    sorted_rects[label].append(rect)

  return centroids, sorted_rects

# returns a rect centered on a centroid with a shape derived from the input rects
def get_representing_rect(centroid, rects, representing_rect_area):
  width_sum = 0
  height_sum = 0
  for rect in rects:
    x1, y1, x2, y2 = rect
    width_sum += abs(x2 - x1)
    height_sum += abs(y2 - y1)
  area = width_sum * height_sum

  # by how much should the area decrease
  area_decrease_factor = area / representing_rect_area
  side_decrease_factor = math.sqrt(area_decrease_factor)

  representing_rect_width = width_sum // side_decrease_factor
  representing_rect_height = height_sum // side_decrease_factor

  center_x, center_y = centroid
  x1 = center_x - representing_rect_width // 2
  y1 = center_y - representing_rect_height // 2
  x2 = x1 + representing_rect_width
  y2 = y1 + representing_rect_height

  return [x1, y1, x2, y2]

# shifts the rectangle into input confines if possible
def confine_to_area(width, height, rect):
  x1, y1, x2, y2 = rect
  if x1 < 0:
    delta = -x1
    x1 += delta
    x2 += delta
  if y1 < 0:
    delta = -y1
    y1 += delta
    y2 += delta
  if x2 > width:
    delta = x2 - width
    x1 -= delta
    x2 -= delta
  if y2 > height:
    delta = y2 - height
    y1 -= delta
    y2 -= delta

  return [x1, y1, x2, y2]

# shifts a rectangle randomly by applying a gaussian noise with standard deviation == magnitude
def random_perturbation(rect, magnitude = 10):
   x1, y1, x2, y2 = rect
   dx = random.gauss(0, magnitude)
   dy = random.gauss(0, magnitude)
   return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

# increases the size of a rectangle randomly by a gaussian noise with standard deviation == magnitude
def random_perturbation_size(rect, magnitude = 10):
   x1, y1, x2, y2 = rect
   dx = random.gauss(1, magnitude)
   dy = random.gauss(1, magnitude)

   center_x = (x1 + x2) / 2
   center_y = (y1 + y2) / 2
   width = x2 - x1
   height = y2 - y1
   new_width = width * dx
   new_height = height * dy

   return [center_x - (new_width/2), center_y - (new_height/2), center_x + (new_width/2), center_y + (new_height/2)]

# shift the rectangle and changes its size
# rect position shifted by a random scalar taken from a gaussian with
#   the perturbation factor multiplied by 100 as the standard deviation
# rect width and height is multiplied by a random factor taken
#   from a gaussian with the perturbation factor as the standard deviation
def pertube_rect(rect, pertubation_factor, frame_width, frame_height):
  shift_factor = pertubation_factor * 100

  pertubed_rect = random_perturbation(rect, shift_factor)
  pertubed_rect = random_perturbation_size(pertubed_rect, pertubation_factor)
  pertubed_rect = normalize(pertubed_rect)
  pertubed_rect = confine_to_area(frame_width, frame_height, pertubed_rect)
  return pertubed_rect

def pertube_rect_multi(rect, pertubation_factor, shift_factor, frame_width, frame_height):
  pertubed_rect = random_perturbation(rect, shift_factor)
  pertubed_rect = random_perturbation_size(pertubed_rect, pertubation_factor)
  pertubed_rect = normalize(pertubed_rect)
  pertubed_rect = confine_to_area(frame_width, frame_height, pertubed_rect)
  return pertubed_rect
