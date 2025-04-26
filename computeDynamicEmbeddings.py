import sys
import datetime

print(f"Job started at: {datetime.datetime.now()}")

import lib.dynamicAnalysisTool as dat

rects = [
  # two rects atop each other
  [0, 0, 50, 100],
  [0, 100, 50, 200],

  # same two rects but shifted 500 pixels to the right
  [500, 0, 550, 100],
  [500, 100, 550, 200],
]

representing_rects = dat.get_representing_rects(rects, 2, 1000)
print(representing_rects)
