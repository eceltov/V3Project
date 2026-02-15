import datetime

print(f"Job started at: {datetime.datetime.now()}")

import lib.dynamicAnalysisTool as dat
import lib.databaseGateway as db

# read detections provided by GroundingDINO
rects = db.read_detection_boxes()

embed_config = {
  "model_year": "2025",
}

# calculate embeddings for all detections (takes a long time)
dat.save_all_dynamic_embeddings(rects, embed_config)
