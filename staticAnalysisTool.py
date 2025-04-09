import torch
import processingTool as pt
import rankCalculations as rc
import boundaries
import iou

# returns a list of tensors, where each tensor represents a section defined by a boundary
def extract_embeddings(model_year, get_boundaries_callback):
  import torch

  model, preprocess, _ = pt.get_model(model_year)
  filenames, _, _, _ = pt.get_MVK_metadata()

  # the callback returns an array of length equal to the section count
  section_count = len(get_boundaries_callback(100, 100))
  section_embeds = [[] for i in range(section_count)]

  with torch.no_grad(), torch.amp.autocast(pt.device):
    for i in range(len(filenames)):
      sections = boundaries.get_image_sections(filenames[i], get_boundaries_callback)

      if i % 100 == 0:
        print("processed images:", i)

      for section_idx in range(section_count):
        preprocessed = preprocess(sections[section_idx]).unsqueeze(0).to(pt.device)
        section_embeds[section_idx].append(model.encode_image(preprocessed).to("cpu"))

  concat_sections = []
  for section_idx in range(section_count):
    concat_sections.append(torch.concat(section_embeds[section_idx]))

  return concat_sections

def save_PraK_embeddings(model_year):
  # boundaries used by PraK
  get_boundaries_callback = boundaries.get_corner_and_centerpiece_overlap_boundaries
  embeddings = extract_embeddings(model_year, get_boundaries_callback)
  pt.write_static_embeddings(model_year, "centerpiece_overlap", embeddings)

def save_whole_embeddings(model_year):
  get_boundaries_callback = boundaries.get_whole_boundaries
  embeddings = extract_embeddings(model_year, get_boundaries_callback)
  pt.write_static_embeddings(model_year, "whole", embeddings)

def get_file_results(file_id, model, tokenizer, embed_config, kind):
  annotations = pt.get_file_annotations(file_id, embed_config["skippable"])
  embeds = pt.read_static_embeddings(embed_config["model_year"], kind).to(pt.device)
  segment_rects = boundaries.get_corner_and_centerpiece_overlap_boundaries(pt.frame_width, pt.frame_height)

  result_list = []
  for annotation in annotations:
    annotation_id = annotation["id"]

    # skip if the embeddings file does not exist
    if not pt.does_derived_dataset_embeddings_file_exist(file_id, annotation_id, embed_config):
      continue

    frame_idx = annotation["frameIdx"]
    frame_rect = annotation["rect"]
    desc_short = annotation["desc_short"]
    desc_long = annotation["desc_long"]
    segment_idx, IoU = iou.get_best_IoU_segment_idx(frame_rect, segment_rects)
  
    rank_short = rc.get_frame_rank(desc_short, frame_idx, embeds[segment_idx], model, tokenizer)
    rank_long = rc.get_frame_rank(desc_long, frame_idx, embeds[segment_idx], model, tokenizer)
    result_list.append({
      "rank_short": rank_short,
      "rank_long": rank_long,
      "IoU": IoU,
    })
    print("#", end="", flush=True)

  return result_list
