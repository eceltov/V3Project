import processingTool as pt
from resultAggregator import ResultAggregator
import optimalGridAnalysisTool as ogat
import staticAnalysisTool as sat

skippable_filenames, _ = pt.get_annotation_filenames_and_dir_path(True)
not_skippable_filenames, _ = pt.get_annotation_filenames_and_dir_path(False)
filenames = {
  True: skippable_filenames,
  False: not_skippable_filenames,
}

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
      print(".", end="", flush=True)
  results.to_csv(pt.static_csv_path)

save_optimal_grid_results()
