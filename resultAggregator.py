import pandas as pd
from pathlib import Path

class ResultAggregator():
  def __init__(self):
    self.results: list[dict] = []

  def append_results(self, result_list):
    for result in result_list:
      self.results.append(result)

  def to_csv(self, filepath):
    # create containing folder
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(self.results)
    df.to_csv(filepath)
