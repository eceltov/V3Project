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

def append_optimal_grid_results(results: ResultAggregator):
    for embed_config in pt.get_optimal_embed_configs():
      model, _, tokenizer = pt.get_model(embed_config["model_year"])
      for file_id in range(len(filenames[embed_config["skippable"]])):
        file_results = ogat.get_file_results(file_id, model, tokenizer, embed_config)
        results.append_file_optimal(embed_config, file_id, file_results)
        print(".", end="", flush=True)

def append_optimal_grid_results(results: ResultAggregator):
  for embed_config in pt.get_static_embed_configs():
    model, _, tokenizer = pt.get_model(embed_config["model_year"])
    for file_id in range(len(filenames[embed_config["skippable"]])):
      file_results = sat.get_file_results(file_id, model, tokenizer, embed_config)
      results.append_file_static(embed_config, file_id, file_results)
      print(".", end="", flush=True)

results = ResultAggregator()
append_optimal_grid_results(results)
print()
print(results)
print(results.results)
