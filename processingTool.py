import json
import os
import pickle
from pathlib import Path
import rectangles

def get_config():
  f = open("./config.json", "r")
  return json.loads(f.read())

config = get_config()
device = config["device"]
dataset_path = config["datasetPath"]
annotations_config = config["annotations"]
annotations_dir_skippable = os.path.join(annotations_config["annotationsDir"], annotations_config["skippableDir"])
annotations_dir_not_skippable = os.path.join(annotations_config["annotationsDir"], annotations_config["notSkippableDir"])
annotation_filenames_skippable = sorted(os.listdir(annotations_dir_skippable))
annotation_filenames_not_skippable = sorted(os.listdir(annotations_dir_not_skippable))
derived_dataset_embeddings_config = config["derivedDatasetEmbeddings"]
static_embeddings_config = config["staticEmbeddings"]
frame_width = config["frameWidth"]
frame_height = config["frameHeight"]
box_enlargement_step = config["boxEnlargementStep"]
optimal_csv_path = derived_dataset_embeddings_config["csvPath"]
static_csv_path = static_embeddings_config["csvPath"]

def get_annotation_filenames_and_dir_path(skippable):
  if skippable:
    annotation_filenames = annotation_filenames_skippable
    annotations_dir_path = annotations_dir_skippable
  else:
    annotation_filenames = annotation_filenames_not_skippable
    annotations_dir_path = annotations_dir_not_skippable
  return annotation_filenames, annotations_dir_path

def annotation_file_exists(file_id, skippable):
  annotation_filenames, annotations_dir_path = get_annotation_filenames_and_dir_path(skippable)
  return len(annotation_filenames) > file_id

# loads all annotation files in ./annotations and joins them into a single annotation list
def get_annotations(skippable):
  annotations = []
  annotation_filenames, annotations_dir_path = get_annotation_filenames_and_dir_path(skippable)

  for filename in annotation_filenames:
    file_path = os.path.join(annotations_dir_path, filename)
    file = open(file_path, "r")
    content = json.loads(file.read())
    annotations += content

  return annotations

def get_file_annotations(file_id, skippable):
  annotation_filenames, annotations_dir_path = get_annotation_filenames_and_dir_path(skippable)
  filename = annotation_filenames[file_id]
  file_path = os.path.join(annotations_dir_path, filename)
  file = open(file_path, "r")
  content = json.loads(file.read())
  return content

def get_first_n_annotations(file_id, n, skippable):
  return get_file_annotations(file_id, skippable)[:n]

def get_annotation(file_id, annotation_id, skippable):
  return get_file_annotations(file_id, skippable)[annotation_id]

def get_filename_from_file_id(file_id, skippable):
  annotation_filenames, annotations_dir_path = get_annotation_filenames_and_dir_path(skippable)
  return annotation_filenames[file_id]


def get_2025_model():
  import open_clip
  import torch

  model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-SO400M-14-SigLIP-384',
    pretrained='webli',
    device=device)
  checkpoint_path = 'models/MCIP-ViT-SO400M-14-SigLIP-384.pth'
  mcip_state_dict = torch.load(checkpoint_path)
  model.load_state_dict(mcip_state_dict, strict=True)
  tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')

  return model, preprocess, tokenizer

def get_2024_model():
  import open_clip

  model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    device=device)
  tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

  return model, preprocess, tokenizer

def get_model(year: str):
  if year == "2024":
    return get_2024_model()
  elif year == "2025":
    return get_2025_model()
  raise KeyError(f"Year '{year}' not found among models.")

metadata_cache = None
# iterates over all frames in the dataset and returns:
# 1. an array of absolute filepaths
# 2. a map from video folders to their frame indices: absolute dirpath => [frame indices]
# 3. a map: frame idx => absolute frame path
# 4. a map: absolute frame path => frame idx
def get_MVK_metadata():
  # caching mechanism so that all files do not have to be iterated over and over
  global metadata_cache

  if metadata_cache != None:
    return metadata_cache

  filepaths = []
  video_to_frame_indices_map = {}
  frame_idx_to_frame_path_map = {}
  frame_path_to_frame_idx_map = {}

  idx = 0
  for dirname in sorted(os.listdir(dataset_path)):
    dirpath = os.path.join(dataset_path, dirname)
    video_indices = []
    for fn in sorted(os.listdir(dirpath)):
      filename = os.path.join(dirpath, fn)
      filepaths.append(filename)
      video_indices.append(idx)
      frame_idx_to_frame_path_map[idx] = filename
      frame_path_to_frame_idx_map[filename] = idx
      idx += 1
    video_to_frame_indices_map[dirpath] = video_indices

  metadata_cache = filepaths, video_to_frame_indices_map, frame_idx_to_frame_path_map, frame_path_to_frame_idx_map
  return metadata_cache

def get_annotation_rect(annotation, embed_config):
  # swap coords so that the first point has lower coords than the second
  x1, y1, x2, y2 = rectangles.normalize(annotation["rect"])

  enlargement = box_enlargement_step * embed_config["box_enlargements"]
  x1 = max(0, x1 - enlargement)
  y1 = max(0, y1 - enlargement)
  x2 = min(frame_width, x2 + enlargement)
  y2 = min(frame_height, y2 + enlargement)
  return [x1, y1, x2, y2]

def write_pickle_file(file_path, data):
  with open(file_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle_file(file_path):
  with open(file_path, 'rb') as handle:
    return pickle.load(handle)

def get_derived_dataset_embeddings_dir(embed_config):
  path = derived_dataset_embeddings_config["mainDir"]
  # whether the annotator could skip frames
  if embed_config["skippable"]:
    path = os.path.join(path, derived_dataset_embeddings_config["skippableDir"])
  else:
    path = os.path.join(path, derived_dataset_embeddings_config["notSkippableDir"])

  # whether the bounding boxes are the original ones drawn by the annotator
  box_enlargements = embed_config["box_enlargements"]
  if box_enlargements == 0:
    path = os.path.join(path, derived_dataset_embeddings_config["originalBoundingBoxDir"])
  else:
    folder_name = derived_dataset_embeddings_config["enlargedBoundingBoxDir"] + str(box_enlargements)
    path = os.path.join(path, folder_name)
    
  # add model year
  path = os.path.join(path, str(embed_config["model_year"]))
  return path  

def get_static_embeddings_filename(model_year, kind):
  return f"{kind}_{model_year}_embeddings"

def get_derived_dataset_embeddings_filename(file_id, annotation_id, skippable):
  annotations_filename = get_filename_from_file_id(file_id, skippable)
  data_filename = f"{annotations_filename}_{annotation_id}"
  return data_filename

def write_derived_dataset_embeddings(file_id, annotation_id, data, embed_config):
  data_filename = get_derived_dataset_embeddings_filename(file_id, annotation_id, embed_config["skippable"])
  # create folder if it does not exist
  derived_dataset_embeddings_dir = get_derived_dataset_embeddings_dir(embed_config)
  Path(derived_dataset_embeddings_dir).mkdir(parents=True, exist_ok=True)
  write_pickle_file(os.path.join(derived_dataset_embeddings_dir, data_filename), data)

def read_derived_dataset_embeddings(file_id, annotation_id, embed_config):
  derived_dataset_embeddings_dir = get_derived_dataset_embeddings_dir(embed_config)
  data_filename = get_derived_dataset_embeddings_filename(file_id, annotation_id, embed_config["skippable"])
  return read_pickle_file(os.path.join(derived_dataset_embeddings_dir, data_filename))

def write_static_embeddings(model_year, kind, data):
  data_filename = get_static_embeddings_filename(model_year, kind)
  # create folder if it does not exist
  static_embeddings_dir = static_embeddings_config["mainDir"]
  Path(static_embeddings_dir).mkdir(parents=True, exist_ok=True)
  write_pickle_file(os.path.join(static_embeddings_dir, data_filename), data)

def read_static_embeddings(model_year, kind):
  data_filename = get_static_embeddings_filename(model_year, kind)
  static_embeddings_dir = static_embeddings_config["mainDir"]
  return read_pickle_file(os.path.join(static_embeddings_dir, data_filename))

# returns the id of the last annotation for the given annotations file
# used to create a derived embeddings file
def get_last_completed_annotation_id(file_id, embed_config):
  annotations_filename = get_filename_from_file_id(file_id, embed_config["skippable"])
  data_filename_prefix = f"{annotations_filename}_"

  # add all annotation ids of the given annotations file
  ids = []
  derived_dataset_embeddings_dir = get_derived_dataset_embeddings_dir(embed_config)

  # check if folder exists
  if not os.path.exists(derived_dataset_embeddings_dir):
    return -1
  
  for filename in os.listdir(derived_dataset_embeddings_dir):
    if filename.startswith(data_filename_prefix):
      annotation_id = filename[len(data_filename_prefix):]
      ids.append(int(annotation_id))
  
  # return -1 if there is no derived embedding file
  if len(ids) == 0:
    return -1
  
  return max(ids)

# returns a list of completed annotation ids for the given annotation file
def get_completed_annotation_ids(file_id, embed_config):
  annotations_filename = get_filename_from_file_id(file_id, embed_config["skippable"])
  data_filename_prefix = f"{annotations_filename}_"

  # add all annotation ids of the given annotations file
  ids = []
  derived_dataset_embeddings_dir = get_derived_dataset_embeddings_dir(embed_config)

  # check if folder exists
  if not os.path.exists(derived_dataset_embeddings_dir):
    return []
  
  for filename in os.listdir(derived_dataset_embeddings_dir):
    if filename.startswith(data_filename_prefix):
      annotation_id = filename[len(data_filename_prefix):]
      ids.append(int(annotation_id))
  
  return ids

def does_derived_dataset_embeddings_file_exist(file_id, annotation_id, embed_config):
  derived_dataset_embeddings_dir = get_derived_dataset_embeddings_dir(embed_config)
  data_filename = get_derived_dataset_embeddings_filename(file_id, annotation_id, embed_config["skippable"])
  path = os.path.join(derived_dataset_embeddings_dir, data_filename)
  return os.path.exists(path)

# creates a key path in a dictionary, if not set already, and returns the leaf dictionary
def make_dict_path(dict: dict, *keys: str):
  for key in keys:
    dict = dict.setdefault(key, {})
  return dict

# returns a list of all evaluated embedding configurations
def get_optimal_embed_configs():
  return derived_dataset_embeddings_config["embedConfigs"]

def get_static_embed_configs():
  return static_embeddings_config["embedConfigs"]
