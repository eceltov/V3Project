import sys
import optimalGridAnalysisTool as ogat
import datetime

print(f"Job started at: {datetime.datetime.now()}")


if len(sys.argv) != 2:
  print("Expected one argument (id of annotation file).")
else:
  embed_config = {
    "skippable": False,
    "box_enlargements": 0,
    "model_year": 2025
  }

  ogat.continue_processing_annotations(int(sys.argv[1]), embed_config)
  embed_config["skippable"] = True
  ogat.continue_processing_annotations(int(sys.argv[1]), embed_config)
  embed_config["model_year"] = 2024
  ogat.continue_processing_annotations(int(sys.argv[1]), embed_config)
  embed_config["skippable"] = False
  ogat.continue_processing_annotations(int(sys.argv[1]), embed_config)

# import processingTool as pt
# model, preprocess, tokenizer = pt.get_2025_model()
# ogat.get_frame_ranks(0, model, tokenizer)
