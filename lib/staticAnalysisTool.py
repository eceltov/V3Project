import torch
import lib.processingTool as pt
import lib.rankCalculations as rc
import lib.boundaries as boundaries
import lib.rectangles as rectangles

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

def kind_to_boundaries_callback(kind):
  if kind == "centerpiece_overlap":
    return boundaries.get_corner_and_centerpiece_overlap_boundaries
  if kind == "whole":
    return boundaries.get_whole_boundaries
  raise LookupError(f"Did not find boundaries callback for kind: ${kind}")

def save_embeddings(model_year, kind):
  # boundaries used by PraK
  get_boundaries_callback = kind_to_boundaries_callback(kind)
  embeddings = extract_embeddings(model_year, get_boundaries_callback)
  pt.write_static_embeddings(model_year, kind, embeddings)

def get_file_results(file_id, model, tokenizer, embed_config):
  annotations = pt.get_file_annotations(file_id, embed_config["skippable"])
  embeds = pt.read_static_embeddings(embed_config["model_year"], embed_config["kind"])
  # load segments to gpu
  embeds = [segment_embeds.to(pt.device) for segment_embeds in embeds]
  segment_rects = kind_to_boundaries_callback(embed_config["kind"])(pt.frame_width, pt.frame_height)

  result_list = []
  for annotation in annotations:
    annotation_id = annotation["id"]

    frame_idx = annotation["frameIdx"]
    frame_rect = annotation["rect"]
    desc_short = annotation["desc_short"]
    desc_long = annotation["desc_long"]
    segment_idx, IoU = rectangles.get_best_IoU_segment_idx(frame_rect, segment_rects)

    rank_short = rc.get_frame_rank(desc_short, frame_idx, embeds[segment_idx], model, tokenizer)
    rank_long = rc.get_frame_rank(desc_long, frame_idx, embeds[segment_idx], model, tokenizer)
    result_list.append({
      "author": pt.get_filename_from_file_id(file_id, embed_config["skippable"])[:-len(".json")],
      "skippable": embed_config["skippable"],
      "annotation_id": annotation_id,
      "kind": embed_config["kind"],
      "model_year": embed_config["model_year"],
      "frame_idx": frame_idx,
      "rank_short": rank_short,
      "rank_long": rank_long,
      "IoU": IoU,
    })
    print("#", end="", flush=True)

  return result_list
