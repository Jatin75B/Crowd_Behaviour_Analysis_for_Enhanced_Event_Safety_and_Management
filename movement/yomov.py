import cv2
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

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        try:
            # Check if there are detections in the current frame
            if results is not None and len(results) > 0:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Calculate flow vectors for each object
                flow_vectors = defaultdict(list)
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 1:
                        prev_x, prev_y = track[-2]
                        curr_x, curr_y = track[-1]
                        flow_vector = np.array([curr_x - prev_x, curr_y - prev_y])
                        flow_vectors[track_id].append(flow_vector)

                # Aggregate flow vectors to find average flow direction
                average_flow_direction = np.zeros(2)
                for flow_vector_list in flow_vectors.values():
                    for flow_vector in flow_vector_list:
                        average_flow_direction += flow_vector
                if len(flow_vectors) > 0:
                    average_flow_direction /= len(flow_vectors)

                # Draw the average flow direction on the frame
                x, y = frame.shape[1] // 2, frame.shape[0] // 2  # Center of the frame
                x_end = int(x + average_flow_direction[0] * 50)
                y_end = int(y + average_flow_direction[1] * 50)
                cv2.arrowedLine(annotated_frame, (x, y), (x_end, y_end), (0, 255, 0), 2)

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
        except:
            print("emptyq")
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
