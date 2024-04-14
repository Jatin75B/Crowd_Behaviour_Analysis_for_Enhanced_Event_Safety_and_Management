import cv2
import numpy as np
from Camera_video_mini import density
from speed_calc_mini import speed_estimation1


# Camera index (usually 0 for the default camera)
camera_index = 0

# Set resize parameters
resize_width = 640  # Change this according to your preference
resize_height = 480  # Change this according to your preference

frame_number = 0
speeds=[]

# Open camera capture
cap = cv2.VideoCapture(camera_index)

# Set capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create window to display analyzed frames
cv2.namedWindow('Analyzed Frame', cv2.WINDOW_NORMAL)

prev=0

# Process each frame from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
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
        
    else:
        heatmap_frame=speed_estimation1(speeds, prev, frame, frame_number)
        prev=frame
    
    combined_image2 = cv2.vconcat([heatmap_frame, heatmap_frame])
    
    combined_image = cv2.hconcat([combined_image1, combined_image2])
    
    frame_number+=1
    
    # Display the analyzed frame
    cv2.imshow('Analyzed Frame', combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release resources
cap.release()
cv2.destroyAllWindows()
