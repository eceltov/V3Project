import json
import os

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
recall_annotations_config = config["recallAnnotations"]
recall_annotations_dir = recall_annotations_config["annotationsDir"]
recall_annotations_filenames = sorted(os.listdir(recall_annotations_dir))
recall_annotation_csv_path = recall_annotations_config["csvPath"]
derived_dataset_embeddings_config = config["derivedDatasetEmbeddings"]
static_embeddings_config = config["staticEmbeddings"]
dynamic_embeddings_config = config["dynamicEmbeddings"]
frame_width = config["frameWidth"]
frame_height = config["frameHeight"]
box_enlargement_step = config["boxEnlargementStep"]
optimal_csv_path = derived_dataset_embeddings_config["csvPath"]
static_csv_path = static_embeddings_config["csvPath"]
static_recall_csv_path = static_embeddings_config["recallCsvPath"]
dynamic_csv_path = dynamic_embeddings_config["csvPath"]
detection_boxes_path = dynamic_embeddings_config["detectionBoxesPath"]
dino_checkpoint_path = dynamic_embeddings_config["dinoCheckpointPath"]
dino_config_path = dynamic_embeddings_config["dinoConfigPath"]
annotation_csv_path = annotations_config["csvPath"]
pertubations_per_annotation = static_embeddings_config["pertubationsPerAnnotation"]

# returns a list of all evaluated embedding configurations
def get_optimal_embed_configs():
  embed_configs = []
  for raw_config in derived_dataset_embeddings_config["embedConfigs"]:
    for box_enlargements in raw_config["box_enlargements"]:
      for skippable in raw_config["skippable"]:
        for model_year in raw_config["model_year"]:
            embed_configs.append({
              "skippable": skippable,
              "box_enlargements": box_enlargements,
              "model_year": model_year,
            })
  return embed_configs

def get_static_embed_configs():
  embed_configs = []
  for raw_config in static_embeddings_config["embedConfigs"]:
    for model_year in raw_config["model_year"]:
      for skippable in raw_config["skippable"]:
        for pertubation_factor in raw_config["pertubation_factor"]:
          for kind in raw_config["kind"]:
            embed_configs.append({
              "skippable": skippable,
              "kind": kind,
              "model_year": model_year,
              "pertubation_factor": pertubation_factor,
            })
  return embed_configs

def get_dynamic_embed_configs():
  embed_configs = []
  for raw_config in dynamic_embeddings_config["embedConfigs"]:
    for model_year in raw_config["model_year"]:
      for skippable in raw_config["skippable"]:
        embed_configs.append({
          "skippable": skippable,
          "model_year": model_year,
        })
  return embed_configs
