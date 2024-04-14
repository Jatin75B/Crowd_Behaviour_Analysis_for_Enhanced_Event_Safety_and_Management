from collections import defaultdict, Counter

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = r"Shopping, People, Crowd, Walking.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Store the direction vectors for each track
direction_vectors = defaultdict(list)

# Number of frames to consider for calculating the most common direction
num_frames_to_consider = 1

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, annotated_frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(annotated_frame, persist=True)

        try:
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
                        curr_point[1])), endpoint, (0, 0, 255), 2)
                    direction_vectors[track_id].append(direction_vector)

            # Calculate the average direction vectors across all tracks
            if len(direction_vectors) > 0:
                all_directions = np.array(
                    [direction for directions in direction_vectors.values() for direction in directions])

                # Split the data into three subsets
                num_directions = 3
                num_samples_per_direction = len(all_directions) // num_directions
                direction_subsets = np.split(all_directions, num_directions)

                for i, direction_subset in enumerate(direction_subsets):
                    average_direction = np.mean(direction_subset, axis=0)
                    print("Average Direction", i+1, ":", average_direction)

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

            # Highlight persons whose direction is not common
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point

                # Calculate relative direction
                if len(track) >= 2:
                    prev_point = track[-2]
                    curr_point = track[-1]
                    direction_vector = np.array(
                        curr_point) - np.array(prev_point)
                    # Normalize the direction vector
                    direction_vector = direction_vector / \
                        np.linalg.norm(direction_vector)

                    # Check if the direction is different from any common direction
                    is_different_direction = True
                    for common_direction in direction_subsets:
                        if np.dot(direction_vector, common_direction) > 0.5:  # Adjust threshold as needed
                            is_different_direction = False
                            break

                    # Highlight the person if the direction is not common
                    if is_different_direction:
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                    (255, 0, 255), 2)

        except:
            print("skip")

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
