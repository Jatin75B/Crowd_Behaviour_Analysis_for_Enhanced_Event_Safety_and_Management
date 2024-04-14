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

# Function to update the GUI with the latest frame
def update_gui():
    global frame_number, speeds, prev
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        return
    
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
    
    # Convert the frame to RGB format for Tkinter
    rgb_frame = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    
    # Update the label with the new image
    label.imgtk = imgtk
    label.config(image=imgtk)
    
    # Schedule the function to be called again
    label.after(1, update_gui)

# Function to close the application
def close_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Open camera capture
cap = cv2.VideoCapture(camera_index)

# Set capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a Tkinter window
root = tk.Tk()
root.title("Camera Feed")

# Create a label widget to display the camera feed
label = tk.Label(root)
label.pack()

# Button to close the application
close_button = tk.Button(root, text="Close", command=close_app)
close_button.pack()

# Start the GUI update loop
update_gui()

# Run the Tkinter event loop
root.mainloop()
