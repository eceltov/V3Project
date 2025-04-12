import processingTool as pt

class ResultAggregator():
  def __init__(self):
    # the dict is structured as follows: Skippable->Enlargements->ModelYear->FileID->AnnotationID->ResultDict
    self.results_optimal: dict[bool, dict[int, dict[str, dict[int, dict[int, dict]]]]] = {}
    self.results_static: dict[bool, dict[str, dict[str, dict[int, dict[int, dict]]]]] = {}

  def append_file_optimal(self, embed_config, file_id, result_dict_list):
    # create path in dict if not present already
    file_dict = pt.make_dict_path(
      self.results_optimal,
      embed_config["skippable"],
      embed_config["box_enlargements"],
      embed_config["model_year"],
    )
    file_dict[file_id] = result_dict_list

  def append_file_static(self, embed_config, file_id, result_dict_list):
    # create path in dict if not present already
    file_dict = pt.make_dict_path(
      self.results_static,
      embed_config["skippable"],
      embed_config["kind"],
      embed_config["model_year"],
    )
    file_dict[file_id] = result_dict_list
