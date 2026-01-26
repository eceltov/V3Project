import sys
import os
import random
import lib.databaseGateway as gateway

try:
    import cv2
except ImportError:
    print("This script requires 'opencv-python'. Please install it using: pip install opencv-python")
    sys.exit(1)

def display_random_frames():
    print("Fetching metadata (this may take a moment)...")
    
    try:
        # get_MVK_metadata returns a tuple: 
        # (filepaths, video_to_frame_indices_map, frame_idx_to_frame_path_map, frame_path_to_frame_idx_map)
        filepaths, _, _, _ = gateway.get_MVK_metadata()
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return

    if not filepaths:
        print("No frames found. Please check your dataset configuration.")
        return

    print(f"Found {len(filepaths)} frames in the dataset.")
    print("Controls:")
    print("  [ENTER]      : Show a new random frame")
    print("  [q] or [ESC] : Quit")

    window_name = "Random Frame Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Initial display
    running = True
    while running:
        # Select a random frame path
        frame_path = random.choice(filepaths)
        
        # Read the image
        image = cv2.imread(frame_path)
        
        if image is None:
            print(f"Failed to load image: {frame_path}")
            # Automatically try the next one if a load fails
            continue
            
        cv2.imshow(window_name, image)
        print(f"Displayed: {frame_path}")
        
        # Wait for user input
        while True:
            key = cv2.waitKey(0)
            
            # Enter key (13 in ASCII) causes loop to break and pick new frame
            if key == 13:
                break
            
            # 'q' or ESC (27) quits the program
            if key == ord('q') or key == 27:
                running = False
                break
                
            # Ignore other keys

    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_random_frames()
