import lib.processingTool as pt
import pandas as pd
from lib.resultAggregator import ResultAggregator
import lib.optimalGridAnalysisTool as ogat
import lib.staticAnalysisTool as sat
import lib.dynamicAnalysisTool as dat
import numpy as np

skippable_filenames, _ = pt.get_annotation_filenames_and_dir_path(True)
not_skippable_filenames, _ = pt.get_annotation_filenames_and_dir_path(False)
filenames = {
  True: skippable_filenames,
  False: not_skippable_filenames,
}

def save_annotations():
  refined_annotations = []
  for skippable, annotation_filenames in filenames.items():
    for file_id in range(len(annotation_filenames)):
      filename = annotation_filenames[file_id]
      annotations: list[dict] = pt.get_file_annotations(file_id, skippable)
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
  df.to_csv(pt.annotation_csv_path, index=False)

def save_optimal_grid_results():
  results = ResultAggregator()
  for embed_config in pt.get_optimal_embed_configs():
    model, _, tokenizer = pt.get_model(embed_config["model_year"])
    for file_id in range(len(filenames[embed_config["skippable"]])):
      file_results = ogat.get_file_results(file_id, model, tokenizer, embed_config)
      results.append_results(file_results)
      print(".", end="", flush=True)
  results.to_csv(pt.optimal_csv_path)

def save_static_grid_results():
  results = ResultAggregator()
  for embed_config in pt.get_static_embed_configs():
    model, _, tokenizer = pt.get_model(embed_config["model_year"])
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
  results.to_csv(pt.static_csv_path)

def save_dynamic_grid_results():
  results = ResultAggregator()
  embed_config = {
    "model_year": "2025",
    "skippable": False,
  }

  ranks = np.load("./rank_index.npy")

  for file_id in range(len(filenames[embed_config["skippable"]])):
    file_results = dat.get_file_results(file_id, embed_config)
    results.append_results(file_results)

  for annotation_idx in range(len(results.results)):
    results.results[annotation_idx]["rank"] = ranks[annotation_idx]
    results.results[annotation_idx]["description_kind"] = "long"

  results.to_csv(pt.dynamic_csv_path)

save_static_grid_results()
