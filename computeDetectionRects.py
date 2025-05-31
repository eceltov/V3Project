import lib.groundingDinoDetector as detector
import lib.configurationProvider as config

detector.compute_rects(config.dino_config_path, config.dino_checkpoint_path)
