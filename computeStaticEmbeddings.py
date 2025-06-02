import sys
import datetime
import lib.staticAnalysisTool as sat

print(f"Job started at: {datetime.datetime.now()}")

# you can use the following kinds: "centerpiece_x", "9_piece_x", "whole"
sat.save_embeddings(model_year="2025", kind="centerpiece_10")
