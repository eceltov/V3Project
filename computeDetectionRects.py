import lib.groundingDINO as detector
import lib.processingTool as pt

detector.compute_rects(pt.dino_config_path, pt.dino_checkpoint_path)
