from PIL import Image
import torch
import torch.nn.functional as F
import datetime
import processingTool as pt
import rankCalculations as rc
import rectangles

def get_frame_section(filename, coords):
  x1, y1, x2, y2 = coords
  frame = Image.open(filename)
  section = frame.crop((x1, y1, x2, y2))
  return section

def get_derived_dataset_embeddings(bounding_box, embed_config):
  model, preprocess, _ = pt.get_model(embed_config["model_year"])
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

def process_annotation(file_id, annotation_id, embed_config):
  annotation = pt.get_annotation(file_id, annotation_id, embed_config["skippable"])
  # apply enlargement to the rect if any
  rect = pt.get_annotation_rect(annotation, embed_config)
  embeddings = get_derived_dataset_embeddings(rect, embed_config)
  pt.write_derived_dataset_embeddings(file_id, annotation_id, embeddings, embed_config)
  print(f"Annotation finished at: {datetime.datetime.now()}")

def process_annotations(annotations, file_id, embed_config):
  for annotation in annotations:
    annotation_id = annotation["id"]
    process_annotation(file_id, annotation_id, embed_config)

def process_first_n_annotations(file_id, n, embed_config):
  print(f"Processing annotations from file {file_id}")
  annotations = pt.get_first_n_annotations(file_id, n, embed_config["skippable"])
  process_annotations(annotations, file_id, embed_config)

def continue_processing_annotations(file_id, embed_config):
  print(f"Processing annotations from file {file_id}")
  if not pt.annotation_file_exists(file_id, embed_config["skippable"]):
    print("Annotation file does not exist")
    return

  # limit to 20 per kind
  last_completed_annotation = pt.get_last_completed_annotation_id(file_id, embed_config)
  if last_completed_annotation >= 19:
    return

  all_annotations = pt.get_file_annotations(file_id, embed_config["skippable"])
  annotations = all_annotations[last_completed_annotation + 1 : 20]
  process_annotations(annotations, file_id, embed_config)
  
def get_file_results(file_id, model, tokenizer, embed_config):
  annotations = pt.get_file_annotations(file_id, embed_config["skippable"])

  result_list = []
  for annotation in annotations:
    annotation_id = annotation["id"]

    # skip if the embeddings file does not exist
    if not pt.does_derived_dataset_embeddings_file_exist(file_id, annotation_id, embed_config):
      continue

    frame_idx = annotation["frameIdx"]
    desc_short = annotation["desc_short"]
    desc_long = annotation["desc_long"]
    rect = pt.get_annotation_rect(annotation, embed_config)
    embeds = pt.read_derived_dataset_embeddings(file_id, annotation_id, embed_config).to(pt.device)
    rank_short = rc.get_frame_rank(desc_short, frame_idx, embeds, model, tokenizer)
    rank_long = rc.get_frame_rank(desc_long, frame_idx, embeds, model, tokenizer)
    result_list.append({
      "rank_short": rank_short,
      "rank_long": rank_long,
      "area": rectangles.get_area(rect),
    })
    print("#", end="", flush=True)

  return result_list
