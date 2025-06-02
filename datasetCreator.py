import lib.databaseGateway as db
import lib.configurationProvider as config
import pandas as pd
from lib.resultAggregator import ResultAggregator
import lib.theoreticalAnalysisTool as tat
import lib.staticAnalysisTool as sat
import lib.dynamicAnalysisTool as dat
import random
from pathlib import Path

# seed the RNG for consistent pertubations
random.seed(0)

skippable_filenames, _ = db.get_annotation_filenames_and_dir_path(True)
not_skippable_filenames, _ = db.get_annotation_filenames_and_dir_path(False)
filenames = {
  True: skippable_filenames,
  False: not_skippable_filenames,
}

# creates a CSV from raw annotations, also filters out duplicates
def save_annotations():
  refined_annotations = []
  for skippable, annotation_filenames in filenames.items():
    for file_id in range(len(annotation_filenames)):
      filename = annotation_filenames[file_id]
      annotations: list[dict] = db.get_file_annotations(file_id, skippable)
      # add extra info and rename columns
      for annotation in annotations:
        annotation["author"] = filename[:-len(".json")]
        annotation["annotation_id"] = annotation["id"]
        annotation.pop("id")
        annotation["frame_idx"] = annotation["frameIdx"]
        annotation.pop("frameIdx")
        annotation["skippable"] = skippable
        refined_annotations.append(annotation)
  df = pd.DataFrame(refined_annotations)
  # rearrange columns
  df = df[["author", "skippable", "annotation_id", "frame_idx", "rect", "desc_short", "desc_long"]]
  # remove duplicated rows (they may have different annotation_id)
  without_annotation_id = df.drop("annotation_id", axis=1)
  df = df.loc[without_annotation_id.astype(str).drop_duplicates().index]
  # create results dir if absent
  Path(config.annotation_csv_path).parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(config.annotation_csv_path, index=False)

def save_theoretical_results():
  results = ResultAggregator()
  for embed_config in config.get_optimal_embed_configs():
    model, _, tokenizer = db.get_model(embed_config["model_year"])
    for file_id in range(len(filenames[embed_config["skippable"]])):
      file_results = tat.get_file_results(file_id, model, tokenizer, embed_config)
      results.append_results(file_results)
      print(".", end="", flush=True)
  results.to_csv(config.optimal_csv_path)

def save_static_grid_results():
  results = ResultAggregator()
  for embed_config in config.get_static_embed_configs():
    model, _, tokenizer = db.get_model(embed_config["model_year"])
    for file_id in range(len(filenames[embed_config["skippable"]])):
      file_results = sat.get_file_results(file_id, model, tokenizer, embed_config)
      results.append_results(file_results)
      # trick to ignore kind of the defined embeds
      if embed_config["kind"] == "whole":
        textual_results_long = sat.get_file_results_textual(file_id, model, tokenizer, embed_config, "long")
        textual_results_short = sat.get_file_results_textual(file_id, model, tokenizer, embed_config, "short")
        results.append_results(textual_results_long)
        results.append_results(textual_results_short)
      print(".", end="", flush=True)
    print("<c>", end="", flush=True)
  results.to_csv(config.static_csv_path)

def save_dynamic_grid_results():
  results = ResultAggregator()
  for embed_config in config.get_dynamic_embed_configs():
    detection_rects = db.read_detection_boxes()
    detection_embeds = db.read_dynamic_embeddings(embed_config["model_year"])
    dat.preprocess_detections(detection_rects, detection_embeds, embed_config)

    model, _, tokenizer = db.get_model(embed_config["model_year"])
    for file_id in range(len(filenames[embed_config["skippable"]])):
      file_results = dat.get_file_results(file_id, detection_rects, detection_embeds, model, tokenizer, embed_config)
      results.append_results(file_results)
      print(".", end="", flush=True)
  results.to_csv(config.dynamic_csv_path)

if __name__ == "__main__":
  # use the various "save_..." functions to produce the datasets you want
  save_annotations()
