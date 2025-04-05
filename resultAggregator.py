import processingTool as pt

class ResultAggregator():
  def __init__(self):
    # the dict is structured as follows: Skippable->Enlargements->ModelYear->FileID->AnnotationID->ResultDict
    self.results: dict[bool, dict[int, dict[str, dict[int, dict[int, dict]]]]]

  def append(self, embed_config, file_id, annotation_id, result_dict):
    # create path in dict if not present already
    annotation_dict = pt.make_dict_path(
      self.results,
      embed_config["skippable"],
      embed_config["box_enlargements"],
      embed_config["model_year"],
      file_id,
    )
    annotation_dict[annotation_id] = result_dict
