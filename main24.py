import cv2
import numpy as np
from Camera_video_mini import density
from speed_calc_mini import speed_estimation1
from collections import defaultdict, Counter
from movement.move_track import track
from ultralytics import YOLO

model=YOLO('yolov8n.pt')

# Function to choose camera or video input
def choose_input():
    while True:
        choice='video'
        #choice = input("Choose input source (camera or video): ").lower()
        if choice == 'camera':
            return 0  # Return camera index
        elif choice == 'video':
            video_path = r'2.mov' #input("Enter video file path: ")
            return video_path  # Return video file path
        else:
            print("Invalid input. Please choose 'camera' or 'video'.")



# Set resize parameters
resize_width = 640  # Change this according to your preference
resize_height = 480  # Change this according to your preference

frame_number = 0
speeds=[]

# Choose input source
input_source = choose_input()

# Open camera capture or video file
if isinstance(input_source, int):  # Camera input
    cap = cv2.VideoCapture(input_source)
else:  # Video file input
    cap = cv2.VideoCapture(input_source)
    
# # Set capture properties
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create window to display analyzed frames
cv2.namedWindow('Analyzed Frame', cv2.WINDOW_NORMAL)

prev=0

# Store the track history
track_history = defaultdict(lambda: [])

# Store the direction vectors for each track
direction_vectors = defaultdict(list)

# Number of frames to consider for calculating the most common direction
num_frames_to_consider = 1

# Process each frame from the camera or video
while True:
    ret, frame = cap.read()
    big_frame=frame
    if not ret:
        print("End of video or failed to capture frame")
        break
    
    # Resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))
    
    overlay=density(frame)
    # Overlay density map on frame
    alpha = 0.5
    
    # Combine frame and overlay
    combined = cv2.addWeighted(frame, 1, overlay, alpha, 0)
    
    combined_image1 = cv2.vconcat([frame, combined])
    if frame_number==0:
        prev=frame
        heatmap_frame=frame
        anotated_frame=frame
        #_,track_history, direction_vectors=track(frame,track_history,direction_vectors,model)
        
    else:
        heatmap_frame=speed_estimation1(speeds, prev, frame, frame_number)
        prev=frame
        #anotated_frame,track_history, direction_vectors=track(big_frame,track_history,direction_vectors,model)
        #anotated_frame = cv2.resize(anotated_frame, (resize_width, resize_height))
    
    
    
    combined_image2 = cv2.vconcat([heatmap_frame, anotated_frame])
    
    combined_image = cv2.hconcat([combined_image1, combined_image2])
    
    frame_number+=1
    
    # Display the analyzed frame
    cv2.imshow('Analyzed Frame', combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release resources
cap.release()
cv2.destroyAllWindows()
