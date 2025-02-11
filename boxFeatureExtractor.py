import jobManager as jm
from detectionBoxes import DetectionBoxes

filepaths, video_to_frame_indices_map, frame_idx_to_frame_path_map, frame_path_to_frame_idx_map = jm.get_images_metadata()

jm.save_clip_box_features(filepaths, "boxFeatures.pickle")


