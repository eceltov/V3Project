import sys
import optimalGridAnalysisTool as ogat
import datetime

print(f"Job started at: {datetime.datetime.now()}")

embed_config = {
  "skippable": False,
  "original_bounding_box": True,
  "model_year": 2025
}

if len(sys.argv) != 2:
  print("Expected one argument (id of annotation file).")
else:
  ogat.continue_processing_annotations(int(sys.argv[1]), embed_config)

# import processingTool as pt
# model, preprocess, tokenizer = pt.get_2025_model()
# ogat.get_frame_ranks(0, model, tokenizer)
