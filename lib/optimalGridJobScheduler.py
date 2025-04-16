import lib.processingTool as pt
import lib.optimalGridAnalysisTool as ogat

class OptimalGridJobScheduler:
  def __init__(self, annotations_per_config, job_id, job_count):
    # how many annotations to process from the file for each configuration
    self.annotations_per_config = annotations_per_config
    self.job_id = job_id
    self.job_count = job_count

    # maps whether the files are skippable to their count
    self.annotation_file_counts: dict[bool, int] = {}

    # dictionary listing remaining tasks (a task is a single annotation for the given configuration)
    # the dict is structured as follows: Skippable->Enlargements->ModelYear->FileID->AnnotationID
    self.all_remaining_tasks: dict[bool, dict[int, dict[str, dict[int, list[int]]]]] = {}
    self.all_tasks: dict[bool, dict[int, dict[str, dict[int, list[int]]]]] = {}

    self.__init_file_counts()
    self.__init_tasks()
    self.__init_all_tasks_flattened()
    self.__init_remaining_tasks_flattened()

  # init the number of annotation files
  def __init_file_counts(self):
    annotation_filenames_skippable, _ = pt.get_annotation_filenames_and_dir_path(True)
    annotation_filenames_not_skippable, _ = pt.get_annotation_filenames_and_dir_path(False)
    self.annotation_file_counts[True] = len(annotation_filenames_skippable)
    self.annotation_file_counts[False] = len(annotation_filenames_not_skippable)

  def __init_tasks(self):
    for embed_config in pt.get_optimal_embed_configs():
      self.__append_tasks(embed_config)

  # appends the remaining_tasks and all_tasks dictionaries with tasks matching the configuration
  def __append_tasks(self, embed_config):
    skippable = embed_config["skippable"]
    for task_dict in [self.all_remaining_tasks, self.all_tasks]:
      if skippable not in task_dict:
        task_dict[skippable] = {}

      enlargements_dict = task_dict[skippable]
      enlargements = embed_config["box_enlargements"]
      if enlargements not in enlargements_dict:
        enlargements_dict[enlargements] = {}

      model_year_dict = enlargements_dict[enlargements]
      model_year = embed_config["model_year"]
      if model_year not in model_year_dict:
        model_year_dict[model_year] = {}

      file_id_dict = model_year_dict[model_year]
      for file_id in range(self.annotation_file_counts[skippable]):
        # list of not completed tasks
        file_id_dict[file_id] = []

        # only append missing tasks
        if task_dict == self.all_remaining_tasks:
          missing_annotations_list = file_id_dict[file_id]
          completed_ids = pt.get_completed_annotation_ids(file_id, embed_config)
          for annotation_id in range(self.annotations_per_config):
            if annotation_id not in completed_ids:
              missing_annotations_list.append(annotation_id)
        # append all tasks
        else:
          annotations_list = file_id_dict[file_id]
          for annotation_id in range(self.annotations_per_config):
            annotations_list.append(annotation_id)

  # creates a flat list of dictionaries containing the embed_config, file_id and annotation_id of the task
  def __init_all_tasks_flattened(self):
    flattened_tasks: list[dict] = []
    for skippable, enlargements_dict in self.all_tasks.items():
      for enlargements, model_year_dict in enlargements_dict.items():
        for model_year, file_id_dict in model_year_dict.items():
          for file_id, missing_annotations_list in file_id_dict.items():
            for missing_annotation_id in missing_annotations_list:
              flattened_tasks.append({
                "embed_config": {
                  "skippable": skippable,
                  "box_enlargements": enlargements,
                  "model_year": model_year
                },
                "file_id": file_id,
                "annotation_id": missing_annotation_id
              })
    self.all_tasks_flattened = flattened_tasks

  def __task_in_all_remaining_tasks(self, task):
    task_embed_config = task["embed_config"]
    if task_embed_config["skippable"] in self.all_remaining_tasks:
      enlargements_dict = self.all_remaining_tasks[task_embed_config["skippable"]]
      if task_embed_config["box_enlargements"] in enlargements_dict:
        model_year_dict = enlargements_dict[task_embed_config["box_enlargements"]]
        if task_embed_config["model_year"] in model_year_dict:
          file_id_dict = model_year_dict[task_embed_config["model_year"]]
          if task["file_id"] in file_id_dict:
            missing_annotations_list = file_id_dict[task["file_id"]]
            return task["annotation_id"] in missing_annotations_list
    return False

  # goes through all tasks assigned to this job and saves those that are not yet completed
  def __init_remaining_tasks_flattened(self):
    flattened_tasks: list[dict] = []
    for task_id in self.get_task_ids():
      task = self.all_tasks_flattened[task_id]
      if self.__task_in_all_remaining_tasks(task):
        flattened_tasks.append(task)
    self.remaining_tasks_flattened = flattened_tasks

  def debug_print_remaining_tasks(self):
    for task in self.remaining_tasks_flattened:
      print(task)

  def debug_print_remaining_task_count(self):
    print(f"Remaining tasks for job_id {self.job_id}: {len(self.remaining_tasks_flattened)}")

  def get_task_ids(self):
    # select every job_count-th task to do
    return range(self.job_id, len(self.all_tasks_flattened), self.job_count)

  # based on how many concurrent jobs are running, select unique tasks based on job_id
  def do_tasks(self):
    for task in self.remaining_tasks_flattened:
      ogat.process_annotation(task["file_id"], task["annotation_id"], task["embed_config"])
