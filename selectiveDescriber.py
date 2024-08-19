import dearpygui.dearpygui as dpg
import os
from PIL import Image
import numpy as np
import random
import math
import jobManager as jm

dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

# whether the user can draw the selection rectangle
selecting = True

screen_width = 1920
screen_height = 1080

img_upscale = 5

img_x = 50
img_y = 200

preview_x = 1350
preview_y = 100

# get images address
filenames, _, _, frame_path_to_frame_idx_map = jm.get_images_metadata()

def image_to_dpg(image):
  image.putalpha(255)
  return np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0

def get_random_dpg_image():
  roll = random.randint(0, len(filenames) - 1)
  image = Image.open(filenames[roll])
  width = image.width * img_upscale
  height = image.height * img_upscale
  image = image.resize((width, height))
  image.putalpha(255)
  dpg_image = np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0

  return (dpg_image, filenames[roll], width, height)

def next_img_shortcut():
  img, filename, img_width, img_height = get_random_dpg_image()

  dpg.set_value("tex", img)
  print(f"showing {filename}")

def next_img_static_shortcut():
  # do not show a new image while the user is typing and presses return by error
  if not selecting:
    return

  dpg.delete_item("img")
  dpg.delete_item("tex")
  
  img, filename, img_width, img_height = get_random_dpg_image()
  with dpg.texture_registry():
    dpg.add_static_texture(width=img_width, height=img_height, default_value=img, tag="tex")
  dpg.add_image("tex", tag="img", pos=[img_x, img_y], parent=window)
  dpg.set_value("filepath", filename)

  print(f"showing {filename}")
  
drawing = False
drawing_start_x = 0
drawing_start_y = 0
drawing_stop_x = 0
drawing_stop_y = 0

def get_image_selection_coords():
  y_err = -4

  x_start = math.floor((drawing_start_x - img_x) / img_upscale)
  y_start = math.floor((drawing_start_y - img_y) / img_upscale) - y_err
  x_end = math.floor((drawing_stop_x - img_x) / img_upscale)
  y_end = math.floor((drawing_stop_y - img_y) / img_upscale) - y_err
  return x_start, y_start, x_end, y_end

def mouse_down_callback(sender, app_data):
  global drawing, drawing_start_x, drawing_start_y

  print("mouse down")

  if not selecting:
    return

  x, y = dpg.get_mouse_pos()
  if not drawing:
    drawing_start_x, drawing_start_y = x, y
    drawing = True
  else:
    dpg.delete_item("rect")
    rect_offset = -10
    dpg.draw_rectangle([drawing_start_x + rect_offset, drawing_start_y - rect_offset], [x + rect_offset, y - rect_offset], tag="rect", thickness=1, color=[255,255,150], parent=window)
    
def get_image_section(filename, coords, upscale=False):
  image = Image.open(filename)
  section = image.crop(coords)

  # upscale the image
  if upscale:
    width = section.width * img_upscale
    height = section.height * img_upscale
    section = section.resize((width, height))

  return section

def mouse_release_callback(sender, app_data):
  global drawing, drawing_stop_x, drawing_stop_y

  print("mouse release")

  if not selecting:
    return

  x, y = dpg.get_mouse_pos()
  drawing_stop_x, drawing_stop_y = x, y
  drawing = False

  x1, y1, x2, y2 = get_image_selection_coords()
  dpg.set_value("selection", f"Selection: [{x1},{y1}] [{x2},{y2}]")

  # set preview
  filename = dpg.get_value("filepath")
  img_section = get_image_section(filename, (x1, y1, x2, y2), upscale=True)
  dpg_img_section = image_to_dpg(img_section)

  # the program crashes when the picture has 0 area
  if abs(x1 - x2) != 0 and abs(y1 - y2) != 0:
    dpg.delete_item("img_section")
    dpg.delete_item("tex_section")
    with dpg.texture_registry():
      dpg.add_static_texture(width=img_section.width, height=img_section.height, default_value=dpg_img_section, tag="tex_section")
    dpg.add_image("tex_section", tag="img_section", pos=[preview_x, preview_y], parent=window)

  print(drawing_start_x, drawing_start_y, drawing_stop_x, drawing_stop_y)
  print(get_image_selection_coords())

def train_callback():
  global selecting

  if not selecting:
    # submit the job and toggle the drawing mode
    text = dpg.get_value("prompt")
    coords = get_image_selection_coords()
    filepath = dpg.get_value("filepath")
    jm.add_job(text, frame_path_to_frame_idx_map[filepath], coords)
    dpg.hide_item("prompt")
  else:
    # focus text mode
    dpg.set_value("prompt", "")
    dpg.show_item("prompt")
    dpg.focus_item("prompt")

  # change text
  selecting = not selecting
  if selecting:
    dpg.set_value("drawing_text", "Drawing ENABLED")
  else:
    dpg.set_value("drawing_text", "Drawing DISABLED")

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=mouse_down_callback)
    dpg.add_mouse_release_handler(callback=mouse_release_callback)

with dpg.texture_registry():
  img, filename, img_width, img_height = get_random_dpg_image()
  dpg.add_static_texture(width=img_width, height=img_height, default_value=img, tag="tex")

with dpg.handler_registry():
  dpg.add_key_press_handler(key=dpg.mvKey_Return, callback=next_img_static_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_F1, callback=train_callback)

with dpg.window(label="Image Window", width=screen_width, height=screen_height, no_collapse=True, no_resize=True, no_close=True, no_move=True, no_title_bar=True, pos=[0, 0]) as window:
  dpg.add_input_text(tag="prompt")
  dpg.add_text("Drawing ENABLED", tag="drawing_text")  
  dpg.add_text("[Return] Show next image")  
  dpg.add_text("[F1] Toggle text/Submit and draw")  
  dpg.add_text(filename, tag="filepath")  
  dpg.add_text(f"Selection: None", tag="selection")  
  dpg.add_image("tex", tag="img", pos=[img_x, img_y])

  dpg.add_text("Selection preview:", pos=(preview_x, preview_y - 20))

dpg.hide_item("prompt")
dpg.show_viewport()
dpg.toggle_viewport_fullscreen()
dpg.start_dearpygui()
dpg.destroy_context()
