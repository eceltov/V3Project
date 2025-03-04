import json
import os
import pickle
from pathlib import Path

def get_config():
  f = open("./config.json", "r")
  return json.loads(f.read())

config = get_config()
device = config["device"]
dataset_path = config["datasetPath"]
annotations_dir_path = config["annotationsDir"]
annotation_filenames = sorted(os.listdir(annotations_dir_path))
derived_dataset_embeddings_dir = config["derivedDatasetEmbeddingsDir"]

# loads all annotation files in ./annotations and joins them into a single annotation list
def get_annotations():
  annotations = []
  # sort files by name to have a total ordering
  for filename in annotation_filenames:
    file_path = os.path.join(annotations_dir_path, filename)
    file = open(file_path, "r")
    content = json.loads(file.read())
    annotations += content

  return annotations

def get_first_n_annotations(file_id, n):
  filename = annotation_filenames[file_id]
  file_path = os.path.join(annotations_dir_path, filename)
  file = open(file_path, "r")
  content = json.loads(file.read())
  return content[:n]

def get_filename_from_file_id(file_id):
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

def write_pickle_file(file_path, data):
  with open(file_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle_file(file_path):
  with open(file_path, 'rb') as handle:
    return pickle.load(handle)

def write_derived_dataset_embeddings(filename, data):
  # create folder if it does not exist
  Path(derived_dataset_embeddings_dir).mkdir(parents=True, exist_ok=True)
  write_pickle_file(os.path.join(derived_dataset_embeddings_dir, filename), data)

def read_derived_dataset_embeddings(filename):
  return read_pickle_file(os.path.join(derived_dataset_embeddings_dir, filename))
