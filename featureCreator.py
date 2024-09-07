import jobManager as jm
import boundaries as b

# creates features for all sections specified by the boundaries of @get_boundaries
def create_features(pickle_filename, get_boundaries):
  filepaths, _, _, _ = jm.get_images_metadata()
  section_count = len(get_boundaries(100, 100)) # trick to get the section_count
  jm.save_clip_section_features(filepaths, pickle_filename, section_count, lambda filename: b.get_image_sections(filename, get_boundaries))

# create_features("whole_features.pickle", b.get_whole_boundaries)
create_features("corner_features.pickle", b.get_corner_boundaries)
create_features("corner_overlap_features.pickle", b.get_corner_overlap_boundaries)
create_features("corner_and_centerpiece_features.pickle", b.get_corner_and_centerpiece_boundaries)
create_features("corner_and_centerpiece_overlap_features.pickle", b.get_corner_and_centerpiece_overlap_boundaries)
create_features("8_piece_overlap_features.pickle", b.get_8_piece_overlap_boundaries)
create_features("8_piece_center_overlap_features.pickle", b.get_8_piece_center_overlap_boundaries)
