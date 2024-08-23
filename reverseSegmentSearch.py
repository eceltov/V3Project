import pickle
import jobManager as jm
import torch.nn.functional as F
import torch
import numpy as np
import time

filepaths, _, frame_idx_to_frame_path_map, frame_path_to_frame_idx_map = jm.get_images_metadata()

# adds a tensor dimension to a 1d tensor (from [512] to [1, 512])
def add_dimension(features_1d):
    return features_1d.view(1, features_1d.size()[0])

# for each segment, the (@reverse_similarity_position)th most similar frame to @frame_path is selected
# then, the most similar frames to those selected frames are found out
# based on these rankings, the best frame is selected which is the most similar to the (@reverse_similarity_position)th most similar frames
def get_segmentation_analysis_best_frame(frame_path, features, reverse_similarity_position):
  segment_count = len(features)
  frame_count = features[0].size()[0]
  similar_distances = []

  frame_idx = frame_path_to_frame_idx_map[frame_path]
  for segment_idx in range(segment_count):
    segment_features = features[segment_idx]

    # adjust the shape of the tensor so that it can be multiplied with the features
    frame_features = add_dimension(segment_features[frame_idx])
    segment_distances = 1 - (F.normalize(frame_features) @ F.normalize(segment_features).T)

    # get sorted indices of best matches to the segment
    sorted_indices = torch.argsort(segment_distances)[0]

    # get the (reverse_similarity_position)th best frame 
    selected_similar_frame = sorted_indices[reverse_similarity_position].item()
    selected_features = add_dimension(segment_features[selected_similar_frame])

    # get distances for the selected similar frame
    selected_distances = 1 - (F.normalize(selected_features) @ F.normalize(segment_features).T)
    similar_distances.append(selected_distances)

  # list of lists of indices of the best matches to the (reverse_similarity_position)th most similar frames 
  sortings = [torch.argsort(similar_distances[i])[0].to("cpu").tolist() for i in range(segment_count)]
  scores = np.zeros(frame_count)

  for segment_idx in range(segment_count):
    segment_sorting = sortings[segment_idx]
    for i in range(frame_count):
      top_i_frame = segment_sorting[i]
      scores[top_i_frame] += i
      
  best_frames = np.argsort(scores)

  return best_frames[0]

  # for i in range(5):
  #   print(frame_idx_to_frame_path_map[best_frames[i]])

# iterates reverse_similarity_position from 0 onward and finds for how long the best frame
# from get_segmentation_analysis_best_frame is the same as the input frame
def find_matching_segmentation_analysis_frame_length(frame_path, features):
  reverse_similarity_position = 0
  while frame_path == frame_idx_to_frame_path_map[get_segmentation_analysis_best_frame(frame_path, features, reverse_similarity_position)]:
    reverse_similarity_position += 1

  return reverse_similarity_position

def get_matching_segmentation_analysis_frame_count(frame_path, features, tests):
  matches = 0
  for i in range(tests):
    if frame_path == frame_idx_to_frame_path_map[get_segmentation_analysis_best_frame(frame_path, features, i)]:
      matches += 1
  return matches

# this section is ugly because this was just to check whether the reverse grid image search works 
frame_paths = [
  "D:\\mvk_resize\\Fiji_Jan2011_0001\\Fiji_Jan2011_0001_049.jpg",
  "D:\\mvk_resize\\Oahu_Jul2022_0015\\Oahu_Jul2022_0015_001.jpg",
  "D:\\mvk_resize\\Oahu_Jul2022_0030\\Oahu_Jul2022_0030_024.jpg",
  "D:\\mvk_resize\\Okinawa_Feb2020_0017\\Okinawa_Feb2020_0017_005.jpg",
  "D:\\mvk_resize\\Padangbai_Jun2022_0054\\Padangbai_Jun2022_0054_022.jpg",
  "D:\\mvk_resize\\Tulamben_Jun2022_0012\\Tulamben_Jun2022_0012_006.jpg",
  "D:\\mvk_resize\\Oahu1_Jul2022_0054\\Oahu1_Jul2022_0054_002.jpg",
  "D:\\mvk_resize\\Oahu2_Jul2022_0035\\Oahu2_Jul2022_0035_006.jpg",
  "D:\\mvk_resize\\Okinawa_Feb2020_0026\\Okinawa_Feb2020_0026_005.jpg",
  "D:\\mvk_resize\\Padangbai_Jun2022_0031\\Padangbai_Jun2022_0031_014.jpg",
  "D:\\mvk_resize\\PhuQuoc_Jun2022_0024\\PhuQuoc_Jun2022_0024_004.jpg",
  "D:\\mvk_resize\\RajaAmpat_Jan2013_0008\\RajaAmpat_Jan2013_0008_003.jpg",
  "D:\\mvk_resize\\Triton_Dec2018_0001\\Triton_Dec2018_0001_001.jpg",
  "D:\\mvk_resize\\Tulamben_Jun2022_0010\\Tulamben_Jun2022_0010_013.jpg",
  "D:\\mvk_resize\\Tulamben_Jun2022_0028\\Tulamben_Jun2022_0028_006.jpg",
  "D:\\mvk_resize\\Tulamben_Jun2022_0048\\Tulamben_Jun2022_0048_003.jpg",
  "D:\\mvk_resize\\Tulamben1_Jun2022_0049\\Tulamben1_Jun2022_0049_001.jpg",
  "D:\\mvk_resize\\Tulamben2_Jun2022_0032\\Tulamben2_Jun2022_0032_012.jpg",
  "D:\\mvk_resize\\Tulamben2_Jun2022_0057\\Tulamben2_Jun2022_0057_022.jpg",
]

feature_paths = [
  "features_laion/corner_features.pickle",
  "features_laion/corner_overlap_features.pickle",
  "features_laion/corner_and_centerpiece_features.pickle",
  "features_laion/corner_and_centerpiece_overlap_features.pickle"
]

feature_list = []
for feature_path in feature_paths:
  with open(feature_path, 'rb') as handle:
    features = [features.to("cuda") for features in pickle.load(handle)]
  feature_list.append(features)

result_sums = [0 for i in range(len(feature_list))]
test_count = 100 # threshold
for frame_path in frame_paths:
  print(frame_path)
  for feature_idx in range(len(feature_paths)):
    # perform the reverse grim image search method for a frame, grid layout (feature), and threshold (test_count)
    matches = get_matching_segmentation_analysis_frame_count(frame_path, feature_list[feature_idx], test_count)
    result_sums[feature_idx] += matches
    print(f"{feature_paths[feature_idx]}: {matches}/{test_count}")
  print()
result_averages = [sum / len(frame_paths) for sum in result_sums]
print(result_averages)
