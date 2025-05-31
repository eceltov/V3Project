import torch
import lib.databaseGateway as db
import lib.configurationProvider as config
import lib.rankCalculations as rc
import lib.boundaries as boundaries
import lib.rectangles as rectangles
import lib.optimalGridAnalysisTool as ogat
import torch.nn.functional as F

def get_representing_rects(rects: list[list], kmeans_k: int, representing_rect_area: int) -> list[list]:
  """Takes a list of rectangles and clusters them using the kmeans algorithm.
      Defines new rectangles centered around kmean clusters with a given area (all such rectangles have the same area).
      These rectangles have a shape derived from the shape of the input rectangles linked to the matching centroid
      (widths and heights are summed and divided by the same factor to match the @representing_rect_area)
      In case the representing rectangles are outside of the frame, they are shifted.

  Args:
      rects (list[list]): A list of [x1, y1, x2, y2] lists representing input rectangles.
      kmeans_k (int): How many clusters should there be.
      representing_rect_area (int): The area (in pixels) of how big each representing rectangle should be.

  Returns:
      list[list]: Returns a list of [x1, y1, x2, y2] lists representing the rectangles centered around kmeans centroids.
      This list is not sorted and can be ordered differently in identical calls.
  """

  # get centroids and linked rects
  centroids, sorted_rects = rectangles.get_centroids_and_sorted_rects(rects, kmeans_k)
  representing_rects = []
  for i in range(kmeans_k):
    # draw a rectangle around the centroid
    representing_rect = rectangles.get_representing_rect(centroids[i], sorted_rects[i], representing_rect_area)
    # move the rectangle so that it is fully in the frame
    moved_representing_rect = rectangles.confine_to_area(config.frame_width, config.frame_height, representing_rect)
    representing_rects.append(moved_representing_rect)
  return representing_rects

# extracts the embeddings of a set of rects for the given frame
def get_dynamic_embeddings(frame_idx, rects: list[list], model, preprocess):
  _, _, frame_idx_to_frame_path_map, _ = db.get_MVK_metadata()
  filepath = frame_idx_to_frame_path_map[frame_idx]

  embeds = []
  with torch.no_grad(), torch.amp.autocast(config.device):
    for bounding_box in rects:
      frame_section = ogat.get_frame_section(filepath, bounding_box)
      preprocessed = preprocess(frame_section).unsqueeze(0).to(config.device)
      embeds.append(model.encode_image(preprocessed).to('cpu'))

  # return empty tensor if there are no detections
  if len(embeds) == 0:
    return torch.tensor([])

  concat = torch.concat(embeds)
  return concat

def save_all_dynamic_embeddings(rects: list[list[list]], embed_config):
  """Calculates all detection embeddings and saves them.

  Args:
      rects (list[list[list]]): A list of a list of rectangles (4 element lists). For each frame in the dataset, a list of detection rectangles. 
      embed_config (_type_): The configuration used for the embeddings.
  """
  model, preprocess, _ = db.get_model(embed_config["model_year"])

  embed_list: list[torch.Tensor] = []
  for frame_idx in range(db.get_frame_count()):
    embed_list.append(get_dynamic_embeddings(frame_idx, rects[frame_idx], model, preprocess))
    if frame_idx % 100 == 0:
      print("processed frames:", frame_idx, flush=True)

  db.write_dynamic_embeddings(embed_config["model_year"], embed_list)

def preprocess_detections(detection_rects: list[list[list]], detection_embeds, embed_config):
  whole_embeds = db.read_static_embeddings(embed_config["model_year"], "whole")[0]
  whole_frame_rect = [0, 0, config.frame_width, config.frame_height]
  for frame_idx in range(db.get_frame_count()):
    # add the whole frame as a fallback for when there are no detections
    detection_rects[frame_idx].append(whole_frame_rect)
    frame_embeds = torch.reshape(whole_embeds[frame_idx], (1, whole_embeds[frame_idx].shape[0]))
    detection_embeds[frame_idx] = torch.cat((detection_embeds[frame_idx], frame_embeds), 0)

def search_detection_boxes(annotation_rect: list, annotation_frame_idx, texts: list[str], detection_rects: list[list[list]], detection_embeds, model, tokenizer):
  """Given the list of detection boxes for each dataset frame, finds the best detection box on each frame, ranks them, and finds the rank of the input frame.

  Args:
      annotation_rect (list): The rectangle drawn by the annotator.
      annotation_frame_idx (_type_): The frame idx of the annotation.
      texts (list[str]): The object descriptions provided by the annotator.
      detection_rects (list[list[list]]): A list of a list of rectangles (4 element lists). For each frame in the dataset, a list of detection rectangles.
      detection_embeds (_type_): A List of 2D tensors that can be indexed in the same way as the detection_rect.
      model (_type_): Model used for similarity search.
      tokenizer (_type_): Tokenizer used for similarity search.

  Returns:
      Returns a list of ranks matching the input texts and how many frames resorted to the fallback.
  """
  selected_box_embeds = []
  fallback_selected_count = 0
  # for each set of detections in a frame, select the best detection box embeds (based on IoU)
  for frame_idx in range(db.get_frame_count()):
    box_idx, IoU = rectangles.get_best_IoU_segment_idx(annotation_rect, detection_rects[frame_idx])
    box_embeds = detection_embeds[frame_idx][box_idx]
    selected_box_embeds.append(box_embeds.view(1, box_embeds.shape[0]))

    # count how many times the whole frame was selected
    if box_idx == len(detection_rects[frame_idx]) - 1:
      fallback_selected_count += 1

  # create tensor from collected box embeds
  selected_box_embeds = torch.concat(selected_box_embeds).to(config.device)

  ranks = []
  for text in texts:
    query = tokenizer(text).to(config.device)

    # find ranks and sort them from best to worst
    with torch.no_grad(), torch.amp.autocast(config.device):
      text_features = model.encode_text(query)
      distances = 1 - (F.normalize(text_features) @ F.normalize(selected_box_embeds).T)
      sorted_indices = torch.argsort(distances)[0].tolist()
      rank = sorted_indices.index(annotation_frame_idx)
      ranks.append(rank)
  return ranks, fallback_selected_count

def get_file_results(file_id, detection_rects, detection_embeds, model, tokenizer, embed_config):
  annotations = db.get_file_annotations(file_id, embed_config["skippable"])

  result_list = []
  for annotation in annotations:
    annotation_id = annotation["id"]

    frame_idx = annotation["frameIdx"]
    frame_rect = annotation["rect"]
    desc_short = annotation["desc_short"]
    desc_long = annotation["desc_long"]

    texts = [desc_short, desc_long]
    ranks, fallback_count = search_detection_boxes(frame_rect, frame_idx, texts, detection_rects, detection_embeds, model, tokenizer)
    result_list.append({
      "author": db.get_filename_from_file_id(file_id, embed_config["skippable"])[:-len(".json")],
      "skippable": embed_config["skippable"],
      "annotation_id": annotation_id,
      "model_year": embed_config["model_year"],
      "frame_idx": frame_idx,
      "rank_short": ranks[0],
      "rank_long": ranks[1],
      "fallback_count": fallback_count,
      "annotation_rect_area": rectangles.get_area(frame_rect),
    })

    print("#", end="", flush=True)
  return result_list
