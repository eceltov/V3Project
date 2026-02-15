from PIL import Image
import torch
import torch.nn.functional as F
import datetime
import lib.databaseGateway as db
import lib.configurationProvider as config
import lib.rankCalculations as rc
import lib.rectangles as rectangles

def get_frame_section(filename, coords):
  x1, y1, x2, y2 = coords
  frame = Image.open(filename)
  section = frame.crop((x1, y1, x2, y2))
  return section

def get_derived_dataset_embeddings(bounding_box, embed_config):
  model, preprocess, _ = db.get_model(embed_config["model_year"])
  filepaths, _, _, _ = db.get_MVK_metadata()

  embeds = []
  with torch.no_grad(), torch.amp.autocast(config.device):
    for i in range(len(filepaths)):
      frame_section = get_frame_section(filepaths[i], bounding_box)
      preprocessed = preprocess(frame_section).unsqueeze(0).to(config.device)
      embeds.append(model.encode_image(preprocessed).to('cpu'))

      if i % 100 == 0:
        print(i)

    concat = torch.concat(embeds)
    return concat

def process_annotation(file_id, annotation_id, embed_config):
  annotation = db.get_annotation(file_id, annotation_id, embed_config["skippable"])
  # skip nonexistent annotations
  if annotation == None:
    return
  # apply enlargement to the rect if any
  rect = db.get_annotation_rect(annotation, embed_config)
  embeddings = get_derived_dataset_embeddings(rect, embed_config)
  db.write_derived_dataset_embeddings(file_id, annotation_id, embeddings, embed_config)
  print(f"Annotation finished at: {datetime.datetime.now()}")

def process_annotations(annotations, file_id, embed_config):
  for annotation in annotations:
    annotation_id = annotation["id"]
    process_annotation(file_id, annotation_id, embed_config)

def process_first_n_annotations(file_id, n, embed_config):
  print(f"Processing annotations from file {file_id}")
  annotations = db.get_first_n_annotations(file_id, n, embed_config["skippable"])
  process_annotations(annotations, file_id, embed_config)

def continue_processing_annotations(file_id, embed_config):
  print(f"Processing annotations from file {file_id}")
  if not db.annotation_file_exists(file_id, embed_config["skippable"]):
    print("Annotation file does not exist")
    return

  # limit to 20 per kind
  last_completed_annotation = db.get_last_completed_annotation_id(file_id, embed_config)
  if last_completed_annotation >= 19:
    return

  all_annotations = db.get_file_annotations(file_id, embed_config["skippable"])
  annotations = all_annotations[last_completed_annotation + 1 : 20]
  process_annotations(annotations, file_id, embed_config)
  
def get_file_results(file_id, model, tokenizer, embed_config):
  annotations = db.get_file_annotations(file_id, embed_config["skippable"])

  result_list = []
  for annotation in annotations:
    annotation_id = annotation["id"]

    # skip if the embeddings file does not exist
    if not db.does_derived_dataset_embeddings_file_exist(file_id, annotation_id, embed_config):
      continue

    frame_idx = annotation["frameIdx"]
    desc_short = annotation["desc_short"]
    desc_long = annotation["desc_long"]
    rect = db.get_annotation_rect(annotation, embed_config)
    embeds = db.read_derived_dataset_embeddings(file_id, annotation_id, embed_config).to(config.device)
    rank_short = rc.get_frame_rank(desc_short, frame_idx, embeds, model, tokenizer)
    rank_long = rc.get_frame_rank(desc_long, frame_idx, embeds, model, tokenizer)
    result_list.append({
      "author": db.get_filename_from_file_id(file_id, embed_config["skippable"])[:-len(".json")],
      "skippable": embed_config["skippable"],
      "annotation_id": annotation_id,
      "box_enlargements": embed_config["box_enlargements"],
      "model_year": embed_config["model_year"],
      "frame_idx": frame_idx,
      "rank_short": rank_short,
      "rank_long": rank_long,
      "area": rectangles.get_area(rect),
    })
    print("#", end="", flush=True)

  return result_list
