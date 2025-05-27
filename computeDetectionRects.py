import lib.groundingDINO as detector

config_path = "/mnt/c/school/videa/dynamicAnnotation/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "/mnt/c/school/videa/dynamicAnnotation/GroundingDINO/weights/groundingdino_swint_ogc.pth"

detector.compute_rects(config_path, checkpoint_path)