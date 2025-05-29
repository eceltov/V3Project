import sys
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

# # get a sample annotation
# annotation = pt.get_annotation(0, 0, False)

# # get frame ranks
# ranks = dat.search_detection_boxes(annotation["rect"], annotation["desc_long"], rects, embed_config)

# print(ranks)


# ### Example detection rectangles clustering ###

# detection_rects = [
#   # two rects atop each other
#   [0, 0, 50, 100],
#   [0, 100, 50, 200],

#   # same two rects but shifted 500 pixels to the right
#   [500, 0, 550, 100],
#   [500, 100, 550, 200],
# ]

# representing_rects = dat.get_representing_rects(detection_rects, 2, 1000)
# print(representing_rects)