import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from Camera_video_mini import density
from speed_calc_mini import speed_estimation1

# Camera index (usually 0 for the default camera)
camera_index = 0

# Set resize parameters
resize_width = 640  # Change this according to your preference
resize_height = 480  # Change this according to your preference

frame_number = 0
speeds=[]

# Function to update the displayed frame
def update_frame():
    global frame_number, speeds, prev
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        return
    
    # Resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))
    
    overlay = density(frame)
    # Overlay density map on frame
    alpha = 0.5
    
    # Combine frame and overlay
    combined = cv2.addWeighted(frame, 1, overlay, alpha, 0)
    
    combined_image1 = cv2.vconcat([frame, combined])
    if frame_number == 0:
        prev = frame
        heatmap_frame = frame
    else:
        heatmap_frame = speed_estimation1(speeds, prev, frame, frame_number)
        prev = frame
    
    combined_image2 = cv2.vconcat([heatmap_frame, heatmap_frame])
    
    combined_image = cv2.hconcat([combined_image1, combined_image2])
    
    frame_number += 1
    
    # Convert the OpenCV frame to Tkinter format
    img = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    
    # Update the label with the new image
    label.imgtk = img_tk
    label.config(image=img_tk)
    
    # Schedule the next update after 10 milliseconds
    label.after(10, update_frame)

# Open camera capture
cap = cv2.VideoCapture(camera_index)

# Set capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resize_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resize_height)

# Create a Tkinter window
root = tk.Tk()
root.title("Analyzed Frame")

# Create a label to display the frames
label = tk.Label(root)
label.pack()

# Start updating the frame
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
