import sys
import optimalGridAnalysisTool as ogat
import os
import json

# if len(sys.argv) != 2:
#   print("Expected one argument (id of annotation file).")
# else:
#   ogat.process_first_n_annotations(int(sys.argv[1]), 20)

filename = "../annotations/annotationsVojta2.json"
# create file if it does not exist

with open(filename, "r+") as file:
  annotations = json.loads(file.read())
  file.seek(0)

  for annotation in annotations:
    annotation["id"] += 37
  
  file.write(json.dumps(annotations))
