import processingTool as pt
from PIL import Image
import torch
import torch.nn.functional as F
import datetime

def get_frame_section(filename, coords):
  # swap coords so that the first point has lower coords than the second
  x1, y1, x2, y2 = coords
  if x1 > x2:
    x1, x2 = x2, x1
  if y1 > y2:
    y1, y2 = y2, y1

  frame = Image.open(filename)
  section = frame.crop((x1, y1, x2, y2))
  return section

def get_derived_dataset_embeddings(bounding_box, embed_config):
  if embed_config["model_year"] == 2025:
    model, preprocess, _ = pt.get_2025_model()
  else:
    model, preprocess, _ = pt.get_2024_model()
  filepaths, _, _, _ = pt.get_MVK_metadata()

  embeds = []
  with torch.no_grad(), torch.amp.autocast(pt.device):
    for i in range(len(filepaths)):
      frame_section = get_frame_section(filepaths[i], bounding_box)
      preprocessed = preprocess(frame_section).unsqueeze(0).to(pt.device)
      embeds.append(model.encode_image(preprocessed).to('cpu'))

      if i % 100 == 0:
        print(i)

    concat = torch.concat(embeds)
    return concat
  
def process_annotations(annotations, file_id, embed_config):
  for annotation in annotations:
    embeddings = get_derived_dataset_embeddings(annotation["rect"], embed_config)
    annotation_id = annotation["id"]
    pt.write_derived_dataset_embeddings(file_id, annotation_id, embeddings, embed_config)
    print(f"Annotation finished at: {datetime.datetime.now()}")

def process_first_n_annotations(file_id, n, embed_config):
  print(f"Processing annotations from file {file_id}")
  annotations = pt.get_first_n_annotations(file_id, n)
  process_annotations(annotations, file_id, embed_config)

def continue_processing_annotations(file_id, embed_config):
  print(f"Processing annotations from file {file_id}")
  last_completed_annotation = pt.get_last_completed_annotation_id(file_id, embed_config)
  all_annotations = pt.get_file_annotations(file_id)
  annotations = all_annotations[last_completed_annotation + 1:]
  process_annotations(annotations, file_id, embed_config)

def get_frame_rank(text, frame_idx, embeds, model, tokenizer):
  query = tokenizer(text).to(pt.device)

  with torch.no_grad(), torch.amp.autocast(pt.device):
    text_embeds = model.encode_text(query)

    distances = 1 - (F.normalize(text_embeds) @ F.normalize(embeds).T)
    sorted_indices = torch.argsort(distances)[0].tolist()
    frame_rank = sorted_indices.index(frame_idx)
    return frame_rank
  
def get_frame_ranks(file_id, model, tokenizer, embed_config):
  annotations = pt.get_file_annotations(file_id)

  ranks_short = []
  ranks_long = []
  for annotation in annotations:
    annotation_id = annotation["id"]

    # skip if the embeddings file does not exist
    if not pt.does_derived_dataset_embeddings_file_exist(file_id, annotation_id, embed_config):
      continue

    frame_idx = annotation["frameIdx"]
    desc_short = annotation["desc_short"]
    desc_long = annotation["desc_long"]
    embeds = pt.read_derived_dataset_embeddings(file_id, annotation_id, embed_config).to(pt.device)
    rank_short = get_frame_rank(desc_short, frame_idx, embeds, model, tokenizer)
    rank_long = get_frame_rank(desc_long, frame_idx, embeds, model, tokenizer)
    print(rank_short, rank_long)
