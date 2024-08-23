from PIL import Image
import math

def get_whole_boundaries(width, height):
  return [
    (0, 0, width, height), # one large segment
  ]

def get_corner_boundaries(width, height):
  return [
    (0, 0, width // 2, height // 2), # left upper
    (0, height // 2, width // 2, height), # left lower
    (width // 2, 0, width, height // 2), # right upper
    (width // 2, height // 2, width, height), # right lower
  ]

def get_corner_overlap_boundaries(width, height):
  # segments have a side of 60 % of the original, instead of 50 %
  seg_width = math.floor(width * 0.6)
  seg_height = math.floor(height * 0.6)

  return [
    (0, 0, seg_width, seg_height), # left upper
    (0, height - seg_height, seg_width, height), # left lower
    (width - seg_width, 0, width, seg_height), # right upper
    (width - seg_width, height - seg_height, width, height), # right lower
  ]

def get_corner_and_centerpiece_boundaries(width, height):
  seg_width = math.floor(width * 0.5)
  seg_height = math.floor(height * 0.5)
  centerpiece_x = math.floor(width * 0.25)
  centerpiece_y = math.floor(height * 0.25)

  return [
    (0, 0, seg_width, seg_height), # left upper
    (0, height - seg_height, seg_width, height), # left lower
    (width - seg_width, 0, width, seg_height), # right upper
    (width - seg_width, height - seg_height, width, height), # right lower
    (centerpiece_x, centerpiece_y, centerpiece_x + seg_width, centerpiece_y + seg_height)
  ]

def get_corner_and_centerpiece_overlap_boundaries(width, height):
  seg_width = math.floor(width * 0.6)
  seg_height = math.floor(height * 0.6)
  centerpiece_x = math.floor(width * 0.2)
  centerpiece_y = math.floor(height * 0.2)

  return [
    (0, 0, seg_width, seg_height), # left upper
    (0, height - seg_height, seg_width, height), # left lower
    (width - seg_width, 0, width, seg_height), # right upper
    (width - seg_width, height - seg_height, width, height), # right lower
    (centerpiece_x, centerpiece_y, centerpiece_x + seg_width, centerpiece_y + seg_height)
  ]

def get_image_sections(filename, get_boundaries):
  image = Image.open(filename)
  width, height = image.size

  boundaries = get_boundaries(width, height)
  sections = [image.crop(boundaries[i]) for i in range(len(boundaries))]
  return sections
