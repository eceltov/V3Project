import sys
import optimalGridAnalysisTool as ogat

if len(sys.argv) != 2:
  print("Expected one argument (id of annotation file).")
else:
  ogat.process_first_n_annotations(int(sys.argv[1]), 20)
