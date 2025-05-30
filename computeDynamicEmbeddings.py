import datetime

print(f"Job started at: {datetime.datetime.now()}")

import lib.dynamicAnalysisTool as dat
import lib.processingTool as pt

# create two dummy detection rectangles per dataset frame
rects = pt.read_detection_boxes()

embed_config = {
  "model_year": "2025",
}

# calculate embeddings for all detections (takes a long time)
dat.save_all_dynamic_embeddings(rects, embed_config)
