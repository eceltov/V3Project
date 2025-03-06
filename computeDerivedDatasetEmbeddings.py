import sys
import optimalGridAnalysisTool as ogat
import datetime

print(f"Job started at: {datetime.datetime.now()}")

if len(sys.argv) != 2:
  print("Expected one argument (id of annotation file).")
else:
  ogat.continue_processing_annotations(int(sys.argv[1]))
