import json
import jobManager as jm
import pickle
import math
import boundaries as b

import sys
sys.path.append('statistics')
import plot

f = open("jobs.json", "r")
jobs = json.loads(f.read())

width, height = 249, 140

def getIntervalPoint(point, lower, upper):
  if point < lower:
    return lower
  if point > upper:
    return upper
  return point

# returns the @rects idx with the biggest overlap with @rect
def getBestCoverageRectIdx(rect, rects):
  bestIdx = -1
  bestArea = -1
  coverage = -1 # 

  rectArea = (rect[2] - rect[0]) * (rect[3] - rect[1])

  for i in range(len(rects)):
    overlapX1 = getIntervalPoint(rect[0], rects[i][0], rects[i][2])
    overlapY1 = getIntervalPoint(rect[1], rects[i][1], rects[i][3])
    overlapX2 = getIntervalPoint(rect[2], rects[i][0], rects[i][2])
    overlapY2 = getIntervalPoint(rect[3], rects[i][1], rects[i][3])
    lenX = overlapX2 - overlapX1
    lenY = overlapY2 - overlapY1
    area = lenX * lenY

    if area > bestArea:
      bestArea = area
      bestIdx = i
      coverage = (area / rectArea * 100) // 1

  return bestIdx, coverage

def getScores(rects, feature_path, top_k_stat):
  import open_clip
  import torch.nn.functional as F

  with open(feature_path, 'rb') as handle:
    rectFeatures = [features.to("cuda") for features in pickle.load(handle)]

  model, _, tokenizer = jm.get_model_preprocess_tokenizer()

  avg_position = 0
  top_k_position = 0
  avg_coverage = 0

  for job in jobs:
    rect = job["rect"]
    id = job["id"]
    bestRectIdx, coverage = getBestCoverageRectIdx(rect, rects)
    frame_position = jm.get_frame_feature_position(job["desc"], job["frameIdx"], rectFeatures[bestRectIdx], model, tokenizer)
    avg_position += frame_position
    avg_coverage += coverage
    if frame_position < top_k_stat:
      top_k_position += 1
    #print(f"id {id}, position {frame_position}, rect {bestRectIdx}, cov {int(coverage)} %")

  avg_position /= len(jobs)
  avg_coverage /= len(jobs)
  print(f"{feature_path}: Average position: {math.floor(avg_position)}, Top {top_k_stat} positions: {top_k_position}/{len(jobs)}, Average coverage {math.floor(avg_coverage)} %")
  return avg_position

def getQueryLocalizedScores(rects, whole_features_path, top_k_stat, query_additions, addition_type):
  import open_clip
  import torch.nn.functional as F

  with open(whole_features_path, 'rb') as handle:
    rectFeatures = [features.to("cuda") for features in pickle.load(handle)]

  model, _, tokenizer = jm.get_model_preprocess_tokenizer()

  avg_position = 0
  top_k_position = 0
  avg_coverage = 0

  for job in jobs:
    rect = job["rect"]
    bestRectIdx, coverage = getBestCoverageRectIdx(rect, rects)

    # modify the query, use the addition that best matches the rect location
    if addition_type == "prefix":
      query = query_additions[bestRectIdx] + job["desc"]
    elif addition_type == "suffix":
      query = job["desc"] + query_additions[bestRectIdx]
    else:
      print(f"Error in getQueryLocalizedScores: Unknown addition_type: {addition_type}")

    # use the modified query with the first (and only) whole_features feature
    frame_position = jm.get_frame_feature_position(query, job["frameIdx"], rectFeatures[0], model, tokenizer)
    avg_position += frame_position
    avg_coverage += coverage
    if frame_position < top_k_stat:
      top_k_position += 1
    #print(f"id {id}, position {frame_position}, rect {bestRectIdx}, cov {int(coverage)} %")

  avg_position /= len(jobs)
  avg_coverage /= len(jobs)
  print(f"{whole_features_path}: Average position: {math.floor(avg_position)}, Top {top_k_stat} positions: {top_k_position}/{len(jobs)}, Average coverage {math.floor(avg_coverage)} %")
  return avg_position

corner_query_prefix_additions = [
  "top left ",
  "bottom left ",
  "top right ",
  "bottom right ",
]

corner_query_suffix_additions = [
  " in the top left of the image",
  " in the bottom left of the image",
  " in the top right of the image",
  " in the bottom right of the image",
]

corner_centerpiece_query_prefix_additions = [
  "top left ",
  "bottom left ",
  "top right ",
  "bottom right ",
  "center "
]

corner_centerpiece_query_suffix_additions = [
  " in the top left of the image",
  " in the bottom left of the image",
  " in the top right of the image",
  " in the bottom right of the image",
  " in the center of the image",
]

# iterates over jobs in jobs.json and the features specified below and prints stats about how the different grids performed
top_k_stat = 100

features_and_boundaries = [
  ('whole_features.pickle', b.get_whole_boundaries(width, height)),
  # ('corner_features.pickle', b.get_corner_boundaries(width, height)),
  # ('corner_overlap_features.pickle', b.get_corner_overlap_boundaries(width, height)),
  # ('corner_and_centerpiece_features.pickle', b.get_corner_and_centerpiece_boundaries(width, height)),
  # ('corner_and_centerpiece_overlap_features.pickle', b.get_corner_and_centerpiece_overlap_boundaries(width, height)),
  # ('8_piece_overlap_features.pickle', b.get_8_piece_overlap_boundaries(width, height)),
  # ('8_piece_center_overlap_features.pickle', b.get_8_piece_center_overlap_boundaries(width, height)),
]

stats = [getScores(boundaries, features, top_k_stat) for features, boundaries in features_and_boundaries]
# print("\nlocalized:")
# query_localized_stats = [
#   getQueryLocalizedScores(b.get_corner_boundaries(width, height), "whole_features.pickle", top_k_stat, corner_query_prefix_additions, "prefix"),
#   getQueryLocalizedScores(b.get_corner_boundaries(width, height), "whole_features.pickle", top_k_stat, corner_query_suffix_additions, "suffix"),
#   getQueryLocalizedScores(b.get_corner_and_centerpiece_boundaries(width, height), "whole_features.pickle", top_k_stat, corner_centerpiece_query_prefix_additions, "prefix"),
#   getQueryLocalizedScores(b.get_corner_and_centerpiece_boundaries(width, height), "whole_features.pickle", top_k_stat, corner_centerpiece_query_suffix_additions, "suffix"),
#   getQueryLocalizedScores(b.get_corner_overlap_boundaries(width, height), "whole_features.pickle", top_k_stat, corner_query_prefix_additions, "prefix"),
#   getQueryLocalizedScores(b.get_corner_overlap_boundaries(width, height), "whole_features.pickle", top_k_stat, corner_query_suffix_additions, "suffix"),
#   getQueryLocalizedScores(b.get_corner_and_centerpiece_overlap_boundaries(width, height), "whole_features.pickle", top_k_stat, corner_centerpiece_query_prefix_additions, "prefix"),
#   getQueryLocalizedScores(b.get_corner_and_centerpiece_overlap_boundaries(width, height), "whole_features.pickle", top_k_stat, corner_centerpiece_query_suffix_additions, "suffix"),
# ]

plot.bar_chart([
  (features_and_boundaries[i][0], stats[i]) for i in range(len(features_and_boundaries))
])
