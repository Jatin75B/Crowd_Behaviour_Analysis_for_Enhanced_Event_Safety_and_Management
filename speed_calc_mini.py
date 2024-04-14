import cv2
import numpy as np

from Speed.classifier import classify_movement

def add_heatmap(frame, mag):
    # Apply threshold to select regions with high optical flow
    thresholded_mag = np.where(mag > 2, mag, 0)

    # Normalize the thresholded magnitude values
    mag_normalized = cv2.normalize(thresholded_mag, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(mag_normalized.astype(np.uint8), cv2.COLORMAP_HOT)
    
    # Resize the heatmap to match the dimensions of the frame
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    # Overlay the heatmap on the frame
    heatmap_overlay = cv2.addWeighted(frame, 0.5, heatmap_resized, 0.5, 0)
    
    return heatmap_overlay

def speed_estimation1(speeds, frame1, frame2, frame_number):
        
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude of the optical flow vectors
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Check if there are any valid magnitudes
    if np.any(mag):
        # Calculate the average speed
        average_speed = np.mean(mag)
        avg_spd=average_speed
        speeds.append(average_speed)
    else:
        speeds.append(np.nan)
    
    # Display the frame with visualizations
    movement_label = classify_movement(avg_spd)
    if movement_label == "Faster Movement":
        cv2.putText(frame2, f'Frame: {frame_number,avg_spd,movement_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
    elif movement_label == "Running":
        cv2.putText(frame2, f'Frame: {frame_number,avg_spd,movement_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)
    else:
        cv2.putText(frame2, f'Frame: {frame_number,avg_spd,movement_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

    heatmap_frame = add_heatmap(frame2, mag)
    return heatmap_frame
        

