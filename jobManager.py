import json
import os
from PIL import Image
import pickle

device = "cuda"

def get_model_preprocess_tokenizer():
  import open_clip

  model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32',
    pretrained='laion2b_s34b_b79k', device=device)
  tokenizer = open_clip.get_tokenizer('ViT-B-32')

  return model, preprocess, tokenizer

def get_config():
  f = open("config.json", "r")
  return json.loads(f.read())

# adds a job to the jobs.json file
def add_job(description, frameIdx, rect):
  f = open("jobs.json", "r+")
  jobs = json.loads(f.read())
  f.seek(0)

  jobs.append({
    "id": len(jobs),
    "frameIdx": frameIdx,
    "desc": description,
    "rect": rect
  })

  f.write(json.dumps(jobs))

# iterates over all images in the dataset and returns:
# 1. an array of absolute filepaths
# 2. a map from video folders to their frame indices: absolute dirpath => [frame indices]
# 3. a map: frame idx => absolute frame path
# 4. a map: absolute frame path => frame idx
def get_images_metadata():
  dataset_path = get_config()["datasetPath"]

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

  return filepaths, video_to_frame_indices_map, frame_idx_to_frame_path_map, frame_path_to_frame_idx_map

def get_image_section(filename, coords):
  image = Image.open(filename)
  section = image.crop(coords)
  return section

# goes through unfinished jobs and creates new features for them
# the features are created from rectangular segments of the dataset images, defined by the "rect" job property
def process_jobs():
  import open_clip
  import torch
  import torch.nn.functional as F

  dataset_path = get_config()["datasetPath"]
  model, preprocess, _ = get_model_preprocess_tokenizer()

  filepaths, _, _, _ = get_images_metadata()
  f = open("jobs.json", "r")
  jobs = json.loads(f.read())

  # find id of the first unprocessed job
  next_job_id = 0
  while (os.path.exists(f"features/{next_job_id}.pickle")):
    next_job_id += 1

  with torch.no_grad(), torch.cuda.amp.autocast():
    for job_id in range(next_job_id, len(jobs)):
      print(f"Processing job {job_id}")

      embeds = []
      for i in range(len(filepaths)):
        img_section = get_image_section(filepaths[i], jobs[job_id]["rect"])

        if i % 100 == 0:
          print("processed images:", i)

        preprocessed = preprocess(img_section).unsqueeze(0).to(device)
        embeds.append(model.encode_image(preprocessed).to("cpu"))
        preprocessed.to("cpu") 

      concat = torch.concat(embeds)

      with open(f"features/{job_id}.pickle", 'wb') as handle:
        pickle.dump(concat, handle, protocol=pickle.HIGHEST_PROTOCOL)

# saves a .pickle file holding a list of feature arrays corresponding to the input image sections
# @pickle_filename: filename of the output file
# @section_count: number of sections the images will be split into, also equal to the number of feature arrays
# @get_image_sections: function which takes a filename and returns a list of image sections
def save_clip_section_features(filenames, pickle_filename, section_count, get_image_sections):
  import torch

  model, preprocess, _ = get_model_preprocess_tokenizer()

  section_features = [[] for i in range(section_count)]

  with torch.no_grad(), torch.cuda.amp.autocast():
    for i in range(len(filenames)):
      sections = get_image_sections(filenames[i])

      if i % 100 == 0:
        print("processed images:", i)

      for section_idx in range(len(section_features)):
        preprocessed = preprocess(sections[section_idx]).unsqueeze(0).to(device)
        section_features[section_idx].append(model.encode_image(preprocessed).to("cpu"))
        preprocessed.to("cpu") 

  concat_sections = []
  for section_idx in range(len(section_features)):
    concat_sections.append(torch.concat(section_features[section_idx]))

  with open(pickle_filename, 'wb') as handle:
    pickle.dump(concat_sections, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_frame_feature_position(text, frame_idx, features, model, tokenizer):
  import torch
  import torch.nn.functional as F

  query = tokenizer(text).to(device)

  with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(query)

    distances = 1 - (F.normalize(text_features) @ F.normalize(features).T)
    sorted_indices = torch.argsort(distances)[0].tolist()
    frame_position = sorted_indices.index(frame_idx)
    return frame_position

# prints stats about jobs, specifically the position of the job frame with the given
# text query in the cropped and whole features
# server just as a quick way to check the performance of the cropped features
def get_job_stats():
  import pickle

  model, _, tokenizer = get_model_preprocess_tokenizer()

  f = open("jobs.json", "r")
  jobs = json.loads(f.read())
  for job in jobs:
    job_id = job["id"]
    features_path = f"features/{job_id}.pickle"
    frame_idx = job["frameIdx"]
    description = job["desc"]

    with open(features_path, 'rb') as handle:
      cropped_features = pickle.load(handle).to(device)
    with open("features/whole_images.pickle", 'rb') as handle:
      whole_features = pickle.load(handle).to(device)

    cropped_frame_position = get_frame_feature_position(description, frame_idx, cropped_features, model, tokenizer)
    print(f"Job {job_id} cropped position: {cropped_frame_position}")

    whole_frame_position = get_frame_feature_position(description, frame_idx, whole_features, model, tokenizer)
    print(f"Job {job_id} whole position: {whole_frame_position}")
