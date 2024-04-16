import cv2
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "E:\Major Project Final\movement\Shopping, People, Crowd, Walking.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
direction_vectors=defaultdict(lambda: [])

time10=0
time1=0
desired_fps = cap.get(cv2.CAP_PROP_FPS)

frame_no=0
skip_var=1

# Loop through the video frames
while cap.isOpened():
    time1=time.time()
    # Read a frame from the video
    success, frame = cap.read()
    frame_no+=1
    if success and (frame_no%skip_var==0 or frame_no==1): 
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        try:
            # Check if there are detections in the current frame
            # Get the boxes and track IDs
            # Check if there are detections in the current frame
            if results is not None and len(results) > 0:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks and calculate direction vectors
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(
                        230, 230, 230), thickness=10)

                    # Calculate relative direction
                    if len(track) >= 2:
                        prev_point = track[-2]
                        curr_point = track[-1]
                        direction_vector = np.array(
                            curr_point) - np.array(prev_point)
                        # Normalize the direction vector
                        direction_vector = direction_vector / \
                            np.linalg.norm(direction_vector)
                        # Scale the direction vector for visualization
                        scaled_vector = direction_vector * 50  # Adjust the scaling factor as needed
                        # Calculate the endpoint of the vector
                        endpoint = (
                            int(curr_point[0] + scaled_vector[0]), int(curr_point[1] + scaled_vector[1]))
                        # Draw the direction vector
                        cv2.arrowedLine(annotated_frame, (int(curr_point[0]), int(
                            curr_point[1])), endpoint, (0, 0, 255), 1)
                        direction_vectors[track_id].append(direction_vector)

                # Calculate the average direction vectors across all tracks
                if len(direction_vectors) > 0:
                    all_directions = np.array(
                        [direction for directions in direction_vectors.values() for direction in directions])

                    # Determine the number of direction subsets
                    num_directions = 3
                    num_samples = len(all_directions)
                    num_samples_per_direction = num_samples // num_directions
                    remainder = num_samples % num_directions

                    # Split the data into direction subsets
                    direction_subsets = np.split(all_directions[:-remainder], num_directions)
                    direction_subsets[-1] = np.concatenate((direction_subsets[-1], all_directions[-remainder:]))

                    for i, direction_subset in enumerate(direction_subsets):
                        average_direction = np.mean(direction_subset, axis=0)
                        print("Average Direction", i+1, ":", average_direction)
                        if np.isnan(average_direction).any():
                                continue
                        # Scale the direction vector for visualization
                        scaled_vector = average_direction * 1000  # Adjust the scaling factor as needed

                        # Calculate the endpoint of the vector
                        frame_center = (annotated_frame.shape[1] // 2,
                                        annotated_frame.shape[0] // 2)
                        endpoint = (frame_center[0] + int(scaled_vector[0]),
                                    frame_center[1] + int(scaled_vector[1]))
                        print("Endpoint", i+1, ":", endpoint)

                        # Draw the common direction vector starting from the frame center
                        cv2.arrowedLine(annotated_frame, frame_center,
                                        endpoint, (255, 0, 0), 2)
                
        except:
            print("emptyq")
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    elif not success and not frame_no%skip_var==0:
        # Break the loop if the end of the video is reached
        break
    elif not success:
        break
    
    time1=time.time()-time1
    print(time1)
    time10+=time1
    if frame_no==11:
        #print(time10)
        # Calculate the average time taken to process 10 frames
        avg_time_per_10_frames = time10 / 10

        # Calculate the frames per second for the last 10 frames
        frames_per_second = 1/desired_fps

        # Calculate the number of frames to skip for real-time processing
        skip_var = int(avg_time_per_10_frames/frames_per_second)
        print(frames_per_second,' ',avg_time_per_10_frames)
        print("Frames to skip for real-time processing:", skip_var)
        print("-----------------------------------")
    print(frame_no)
        
        

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
