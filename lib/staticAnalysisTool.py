import torch
import lib.databaseGateway as db
import lib.configurationProvider as config
import lib.rankCalculations as rc
import lib.boundaries as boundaries
import lib.rectangles as rectangles

# returns a list of tensors, where each tensor represents a section defined by a boundary
def extract_embeddings(model_year, get_boundaries_callback):
  import torch

  model, preprocess, _ = db.get_model(model_year)
  filenames, _, _, _ = db.get_MVK_metadata()

  # the callback returns an array of length equal to the section count
  section_count = len(get_boundaries_callback(100, 100))
  section_embeds = [[] for i in range(section_count)]

  with torch.no_grad(), torch.amp.autocast(config.device):
    for i in range(len(filenames)):
      sections = boundaries.get_image_sections(filenames[i], get_boundaries_callback)

      if i % 100 == 0:
        print("processed images:", i, flush=True)

      for section_idx in range(section_count):
        preprocessed = preprocess(sections[section_idx]).unsqueeze(0).to(config.device)
        section_embeds[section_idx].append(model.encode_image(preprocessed).to("cpu"))

  concat_sections = []
  for section_idx in range(section_count):
    concat_sections.append(torch.concat(section_embeds[section_idx]))

  return concat_sections

def kind_to_boundaries_callback(kind):
  tokens = kind.split("_")
  if tokens[0] == "centerpiece":
    return lambda width, height: boundaries.get_corner_and_centerpiece_overlap_boundaries(width, height, int(tokens[1]) / 100)
  if tokens[0] == "9" and tokens[1] == "piece":
    return lambda width, height: boundaries.get_9_piece_overlap_boundaries(width, height, int(tokens[2]) / 100)
  if kind == "whole":
    return boundaries.get_whole_boundaries
  raise LookupError(f"Did not find boundaries callback for kind: ${kind}")

def save_embeddings(model_year, kind):
  # boundaries used by PraK
  get_boundaries_callback = kind_to_boundaries_callback(kind)
  embeddings = extract_embeddings(model_year, get_boundaries_callback)
  db.write_static_embeddings(model_year, kind, embeddings)

def get_file_results(file_id, model, tokenizer, embed_config):
  annotations = db.get_file_annotations(file_id, embed_config["skippable"])
  embeds = db.read_static_embeddings(embed_config["model_year"], embed_config["kind"])
  # load segments to gpu
  embeds = [segment_embeds.to(config.device) for segment_embeds in embeds]
  segment_rects = kind_to_boundaries_callback(embed_config["kind"])(config.frame_width, config.frame_height)

  result_list = []
  for annotation in annotations:
    annotation_id = annotation["id"]

    frame_idx = annotation["frameIdx"]
    frame_rect = annotation["rect"]
    desc_short = annotation["desc_short"]
    desc_long = annotation["desc_long"]
    pertubation = embed_config["pertubation_factor"]

    for perturbation_id in range(config.pertubations_per_annotation):
      # pertube the annotation rectangle randomly (simulation imperfect user input rect)
      if pertubation > 0:
        frame_rect = rectangles.pertube_rect(frame_rect, pertubation, config.frame_width, config.frame_height)

      segment_idx, IoU = rectangles.get_best_IoU_segment_idx(frame_rect, segment_rects)

      rank_short = rc.get_frame_rank(desc_short, frame_idx, embeds[segment_idx], model, tokenizer)
      rank_long = rc.get_frame_rank(desc_long, frame_idx, embeds[segment_idx], model, tokenizer)
      result_list.append({
        "author": db.get_filename_from_file_id(file_id, embed_config["skippable"])[:-len(".json")],
        "skippable": embed_config["skippable"],
        "annotation_id": annotation_id,
        "perturbation_id": perturbation_id,
        "kind": embed_config["kind"],
        "model_year": embed_config["model_year"],
        "frame_idx": frame_idx,
        "rank_short": rank_short,
        "rank_long": rank_long,
        "IoU": IoU,
        "pertubation_factor": pertubation,
      })

      # do not create duplicate rows for unpertubed results
      if pertubation <= 0:
        break

    print("#", end="", flush=True)

  return result_list

suffixes = {
  "short": [
    " in the upper left",
    " in the lower left",
    " in the upper right",
    " in the lower right",
    " in the center",
  ],
  "long": [
    " in the upper left part of the image",
    " in the lower left part of the image",
    " in the upper right part of the image",
    " in the lower right part of the image",
    " in the center part of the image",
  ],
}

def get_centerpiece_overlap_textual_suffix(segment_idx, suffix_kind):
  return suffixes[suffix_kind][segment_idx]

def get_file_results_textual(file_id, model, tokenizer, embed_config, suffix_kind):
  annotations = db.get_file_annotations(file_id, embed_config["skippable"])
  embeds = db.read_static_embeddings(embed_config["model_year"], "whole")
  # load segments to gpu
  embeds = [segment_embeds.to(config.device) for segment_embeds in embeds]
  segment_rects = kind_to_boundaries_callback("centerpiece_10")(config.frame_width, config.frame_height)

  result_list = []
  for annotation in annotations:
    annotation_id = annotation["id"]

    frame_idx = annotation["frameIdx"]
    frame_rect = annotation["rect"]
    segment_idx, IoU = rectangles.get_best_IoU_segment_idx(frame_rect, segment_rects)
    suffix = get_centerpiece_overlap_textual_suffix(segment_idx, suffix_kind)
    desc_short = annotation["desc_short"] + suffix
    desc_long = annotation["desc_long"] + suffix

    rank_short = rc.get_frame_rank(desc_short, frame_idx, embeds[0], model, tokenizer)
    rank_long = rc.get_frame_rank(desc_long, frame_idx, embeds[0], model, tokenizer)
    result_list.append({
      "author": config.get_filename_from_file_id(file_id, embed_config["skippable"])[:-len(".json")],
      "skippable": embed_config["skippable"],
      "annotation_id": annotation_id,
      "perturbation_id": 0,
      "kind": f"textual_{suffix_kind}",
      "model_year": embed_config["model_year"],
      "frame_idx": frame_idx,
      "rank_short": rank_short,
      "rank_long": rank_long,
      "IoU": IoU,
      "pertubation_factor": 0,
    })
    print("#", end="", flush=True)

  return result_list
