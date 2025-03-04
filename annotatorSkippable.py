import dearpygui.dearpygui as dpg
import os
from PIL import Image
import numpy as np
import random
import math
import json

def get_images_metadata():
  f = open("config.json", "r")
  config = json.loads(f.read())
  dataset_path = config["datasetPath"]

  filepaths = []
  video_to_frame_indices_map = {}
  frame_idx_to_frame_path_map = {}
  frame_path_to_frame_idx_map = {}

  idx = 0
  for dirname in sorted(os.listdir(dataset_path)):
    dirpath = os.path.join(dataset_path, dirname)
    video_indices = []
    for fn in sorted(os.listdir(dirpath)):
      filename = os.path.join(dirpath, fn)
      filepaths.append(filename)
      video_indices.append(idx)
      frame_idx_to_frame_path_map[idx] = filename
      frame_path_to_frame_idx_map[filename] = idx
      idx += 1
    video_to_frame_indices_map[dirpath] = video_indices

  return filepaths, video_to_frame_indices_map, frame_idx_to_frame_path_map, frame_path_to_frame_idx_map

filenames, _, _, frame_path_to_frame_idx_map = get_images_metadata()

def get_image_size():
  image = Image.open(filenames[0])
  width = image.width
  height = image.height
  return width, height

dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

screen_width = 1920
screen_height = 1080

# cursor position errors due to paddings and margins
cursor_x_error = -3
cursor_y_error = -2
# img position errors due to paddings and margins
img_y_error = 18
img_x_error = -2

img_upscale = 2

img_x = 20
img_y = 200
# positions without errors
img_x_norm = img_x - img_x_error
img_y_norm = img_y - img_y_error

base_img_width, base_img_height = get_image_size()
scaled_img_width = base_img_width * img_upscale
scaled_img_height = base_img_height * img_upscale

preview_x = 1400
preview_y = 200
max_preview_width = screen_width - preview_x - 20
max_preview_height = screen_height - preview_y - 20

def mouse_pos():
  x, y = dpg.get_mouse_pos()
  x -= cursor_x_error
  y -= cursor_y_error
  return x, y

def image_to_dpg(image):
  image.putalpha(255)
  return np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0

def get_random_dpg_image():
  roll = random.randint(0, len(filenames) - 1)
  image = Image.open(filenames[roll])
  image = image.resize((scaled_img_width, scaled_img_height))
  image.putalpha(255)
  dpg_image = np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0

  return (dpg_image, filenames[roll])

def next_img_shortcut():
  img, filename = get_random_dpg_image()

  dpg.set_value("tex", img)
  print(f"showing {filename}")
  
drawing = False
drawing_start_x = 0
drawing_start_y = 0
drawing_stop_x = 0
drawing_stop_y = 0

def get_image_selection_coords():
  x_err = 1
  y_err = -9

  x_start = math.floor((drawing_start_x - img_x) / img_upscale) - x_err
  y_start = math.floor((drawing_start_y - img_y) / img_upscale) - y_err
  x_end = math.floor((drawing_stop_x - img_x) / img_upscale) - x_err
  y_end = math.floor((drawing_stop_y - img_y) / img_upscale) - y_err
  return x_start, y_start, x_end, y_end

def confine_to_boundaries(value, min, max):
  if value < min:
    return min
  elif value > max:
    return max
  return value

def draw_rect(tag, color):
  dpg.delete_item(tag)
  rect_offset = -10
  dpg.draw_rectangle(
    [drawing_start_x + rect_offset, drawing_start_y - rect_offset],
    [drawing_stop_x + rect_offset, drawing_stop_y - rect_offset],
    tag=tag,
    thickness=2,
    color=color,
    parent=window
  )

def mouse_down_callback(sender, app_data):
  global drawing, drawing_start_x, drawing_start_y, drawing_stop_x, drawing_stop_y

  # boundaries for the cursor
  min_x = img_x_norm
  max_x = img_x_norm + scaled_img_width
  min_y = img_y_norm
  max_y = img_y_norm + scaled_img_height

  x, y = mouse_pos()
  # check whether the mouse is inside the bounds
  if not drawing and (x < min_x or x >= max_x or y < img_y_norm or y >= max_y):
    return
  
  if not drawing:
    drawing_start_x, drawing_start_y = x, y
    drawing = True
  else:
    # if the mouse is outside the bounds, confine it within the image
    drawing_stop_x = confine_to_boundaries(x, min_x, max_x)
    drawing_stop_y = confine_to_boundaries(y, min_y, max_y)
    draw_rect("rect", [255,255,150])
    
def get_image_section(filename, coords, upscale=False):
  # swap coords so that the first point has lower coords than the second
  x1, y1, x2, y2 = coords
  if x1 > x2:
    x1, x2 = x2, x1
  if y1 > y2:
    y1, y2 = y2, y1
  
  image = Image.open(filename)
  section = image.crop((x1, y1, x2, y2))

  width = section.width
  height = section.height

  # do nothing if the dimensions are invalid
  if width < 1 or height < 1:
    return None

  # upscale the image
  if upscale:
    width *= img_upscale
    height *= img_upscale

  # resize the preview to fit in the screen
  reduction_factor = 1
  if width > max_preview_width:
    reduction_factor = max_preview_width / width
  if max_preview_height / height < reduction_factor:
    reduction_factor = max_preview_height / height

  section = section.resize((math.floor(width * reduction_factor), math.floor(height * reduction_factor)))
  return section

def mouse_release_callback(sender, app_data):
  global drawing, drawing_stop_x, drawing_stop_y

  drawing = False

  x1, y1, x2, y2 = get_image_selection_coords()
  dpg.set_value("selection", f"Selection: [{x1},{y1}] [{x2},{y2}]")

  # set preview
  filename = dpg.get_value("filepath")
  img_section = get_image_section(filename, (x1, y1, x2, y2), upscale=True)
  if img_section != None:
    dpg_img_section = image_to_dpg(img_section)

  # the program crashes when the picture has 0 area
  if abs(x1 - x2) != 0 and abs(y1 - y2) != 0:
    dpg.delete_item("img_section")
    dpg.delete_item("tex_section")
    if img_section != None:
      with dpg.texture_registry():
        dpg.add_static_texture(width=img_section.width, height=img_section.height, default_value=dpg_img_section, tag="tex_section")
    dpg.add_image("tex_section", tag="img_section", pos=[preview_x, preview_y], parent=window)

  print(drawing_start_x, drawing_start_y, drawing_stop_x, drawing_stop_y)
  print(get_image_selection_coords())

# adds an annotation to the json file
def save_annotation(short, long, frameIdx, rect):
  filename = "annotations.json"
  # create file if it does not exist
  if not os.path.exists(filename):
    with open(filename, 'w') as file:
      json.dump([], file)

  with open(filename, "r+") as file:
    annotations = json.loads(file.read())
    file.seek(0)

    annotations.append({
      "id": len(annotations),
      "frameIdx": frameIdx,
      "desc_short": short,
      "desc_long": long,
      "rect": rect
    })

    file.write(json.dumps(annotations))
    dpg.set_value("annotations", f"Annotations: {len(annotations)}")

img_annotations = 0

def reset_annotation_state():
  global drawing, drawing_start_x, drawing_start_y, drawing_stop_x, drawing_stop_y

  drawing = False
  drawing_start_x = 0
  drawing_stop_x = 0
  drawing_start_y = 0
  drawing_stop_y = 0
  dpg.set_value("prompt_short", "")
  dpg.set_value("prompt_long", "")

# submit the annotation
def submit_callback():
  global img_annotations

  if drawing_start_x - drawing_stop_x == 0 or drawing_start_y - drawing_stop_y == 0:
    dpg.set_value("error", "Please draw a rectangle with a positive surface area.")
    return


  text_short = dpg.get_value("prompt_short")
  text_long = dpg.get_value("prompt_long")
  if len(text_short) == 0 or len(text_long) == 0:
    dpg.set_value("error", "Please fill out both the short and long description.")
    return

  dpg.set_value("error", "")
  coords = get_image_selection_coords()
  filepath = dpg.get_value("filepath")
  save_annotation(text_short, text_long, frame_path_to_frame_idx_map[filepath], coords)
  dpg.delete_item("rect")
  draw_rect(f"rect{img_annotations}", [0, 0, 0])
  img_annotations += 1
  reset_annotation_state()

def next_img_shortcut():
  reset_annotation_state()

  dpg.delete_item("img")
  dpg.delete_item("tex")

  # remove rectangles
  dpg.delete_item("rect")
  for i in range(img_annotations):
    dpg.delete_item(f"rect{i}")
  
  img, filename = get_random_dpg_image()
  with dpg.texture_registry():
    dpg.add_static_texture(width=scaled_img_width, height=scaled_img_height, default_value=img, tag="tex")
  dpg.add_image("tex", tag="img", pos=[img_x, img_y], parent=window)
  dpg.set_value("filepath", filename)

  print(f"showing {filename}")

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=mouse_down_callback)
    dpg.add_mouse_release_handler(callback=mouse_release_callback)

with dpg.texture_registry():
  img, filename = get_random_dpg_image()
  dpg.add_static_texture(width=scaled_img_width, height=scaled_img_height, default_value=img, tag="tex")

with dpg.handler_registry():
  dpg.add_key_press_handler(key=dpg.mvKey_F12, callback=next_img_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_Return, callback=submit_callback)

with dpg.window(label="Image Window", width=screen_width, height=screen_height, no_collapse=True, no_resize=True, no_close=True, no_move=True, no_title_bar=True, pos=[0, 0]) as window:
  dpg.add_input_text(hint="short description", tag="prompt_short")
  dpg.add_input_text(hint="a longer and more detailed description of the selected object", tag="prompt_long")
  dpg.add_button(label="[Return] Save Annotation", tag="save", callback=submit_callback)
  dpg.add_button(label="[F12] Next Frame", tag="next", callback=next_img_shortcut)
  dpg.add_text("", tag="annotations")  
  dpg.add_text(filename, tag="filepath")  
  dpg.add_text(f"Selection: None", tag="selection")  
  dpg.add_text("", tag="error", color=[255, 120, 120])  
  dpg.add_image("tex", tag="img", pos=[img_x, img_y])

  dpg.add_text("Selection preview:", pos=(preview_x, preview_y - 20))

dpg.show_viewport()
dpg.toggle_viewport_fullscreen()
dpg.start_dearpygui()
dpg.destroy_context()
