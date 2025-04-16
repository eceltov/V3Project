import sys
import datetime
import lib.staticAnalysisTool as sat

print(f"Job started at: {datetime.datetime.now()}")

sat.save_embeddings("2024", "whole")
