import processingTool as pt
from PIL import Image
import torch

def get_frame_section(filename, coords):
  # swap coords so that the first point has lower coords than the second
  x1, y1, x2, y2 = coords
  if x1 > x2:
    x1, x2 = x2, x1
  if y1 > y2:
    y1, y2 = y2, y1

  frame = Image.open(filename)
  section = frame.crop((x1, y1, x2, y2))
  return section

def get_derived_dataset_embeddings(bounding_box):
  model, preprocess, _ = pt.get_2025_model()
  filepaths, _, _, _ = pt.get_MVK_metadata()

  embeds = []
  with torch.no_grad(), torch.amp.autocast(pt.device):
    for i in range(len(filepaths)):
      frame_section = get_frame_section(filepaths[i], bounding_box)
      preprocessed = preprocess(frame_section).unsqueeze(0).to(pt.device)
      embeds.append(model.encode_image(preprocessed).to('cpu'))

      if i % 100 == 0:
        print(i)

    concat = torch.concat(embeds)
    return concat

def process_first_n_annotations(file_id, n):
  annotations_filename = pt.get_filename_from_file_id(file_id)
  print(f"Processing annotations from {annotations_filename}")
  
  annotations = pt.get_first_n_annotations(file_id, n)
  for annotation in annotations:
    embeddings = get_derived_dataset_embeddings(annotation["rect"])
    annotation_id = annotation["id"]
    data_filename = f"{annotations_filename}_{annotation_id}"
    pt.write_derived_dataset_embeddings(data_filename, embeddings)
