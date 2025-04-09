import processingTool as pt

def get_IoU(rect1, rect2):
  # normalize coords
  rect1 = pt.normalize_rectangle(rect1)
  rect2 = pt.normalize_rectangle(rect2)

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
