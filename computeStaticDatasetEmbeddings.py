import sys
import datetime
import staticAnalysisTool as sat

print(f"Job started at: {datetime.datetime.now()}")

sat.save_whole_embeddings("2025")
