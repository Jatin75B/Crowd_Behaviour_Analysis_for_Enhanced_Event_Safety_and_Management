import cv2
import numpy as np
from ultralytics import YOLO
from pykalman import KalmanFilter  # Import KalmanFilter from scikit-learn

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture("qwe.mp4")  # Replace with your video path

# Define parameters for Kalman Filter (adjust based on your video data)
process_noise_cov = 1e-5  # Process noise covariance
measurement_noise_cov = 1e-1  # Measurement noise covariance

# Initialize Kalman Filter (state vector: [x, y, dx/dt, dy/dt])
dt = 1.0  # Time step between frames (assuming constant)
kf = KalmanFilter(state_transition_matrix=np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]], dtype=np.float32),
                   process_noise_cov=np.diag([process_noise_cov, process_noise_cov, 1e-4, 1e-4]),
                   measurement_noise_cov=np.diag([measurement_noise_cov, measurement_noise_cov]),
                   covariance_type='diag')

# Function to convert YOLO detections (xywh format) to Kalman Filter state vector
def convert_to_kf_state(detection):
    x, y, w, h = detection
    cx = x + w / 2
    cy = y + h / 2
    vx = 0  # Initializing velocity to 0 (assuming constant for now)
    vy = 0
    return np.array([cx, cy, vx, vy], dtype=np.float32)

# Function to update Kalman Filter based on detection and predict next state
def update_kalman_filter(kf, detection):
    kf.predict()
    if detection is not None:
        kf.update(convert_to_kf_state(detection))
    return kf.x

# Function to track objects using optical flow and Kalman Filter
def track_objects(prev_frame, curr_frame, prev_boxes):
    tracked_boxes = []
    flow = cv2.optflow.createFarnebackOpticalFlow()  # Create Farneback optical flow object

    # Calculate optical flow
    flow = flow.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 5, 1.2, 0, 15, 0, 1.5, 1.2, 0)

    for i, box in enumerate(prev_boxes):
        # Get Kalman Filter prediction for current frame
        predicted_state = update_kalman_filter(kf[i], box)

        # Extract flow information within the bounding box region
        center_x, center_y = int(predicted_state[0]), int(predicted_state[1])
        flow_x, flow_y = flow[center_y, center_x][0], flow[center_y, center_x][1]

        # Combine Kalman Filter prediction with optical flow for more robust tracking
        tracked_x = int(center_x + flow_x)
        tracked_y = int(center_y + flow_y)

        # Update Kalman Filter covariance based on flow uncertainty (optional)
        # ... (consider adjusting process noise covariance based on flow magnitude)

        # Extract width and height from original detection (assuming they remain constant)
        w, h = box[2], box[3]

        tracked_boxes.append([tracked_x - w // 2, tracked_y - h // 2, w, h])

    return tracked_boxes

# Main loop for video processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection with YOLOv8
    results = model(frame)
    boxes = results.pandas().xyxy[0]  # Assuming single class detection

        # Initialize Kalman Filters for new detections (if any)
    if len(boxes) > 0:
        for i in range(len(prev_boxes), len(boxes)):
            kf_new = KalmanFilter(state_transition_matrix=np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32),
                                 process_noise_cov=np.diag([process_noise_cov, process_noise_cov, 1e-4, 1e-4]),
                                 measurement_noise_cov=np.diag([measurement_noise_cov, measurement_noise_cov]),
                                 covariance_type='diag')
            kf_new.x = convert_to_kf_state(boxes.iloc[i])
            kf.append(kf_new)  # Append new Kalman Filter to the list

    # Track objects using optical flow and Kalman Filter
    tracked_boxes = track_objects(prev_frame, frame, prev_boxes)

    # Draw bounding boxes and labels on the frame
    for box in tracked_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Assuming 'names' attribute holds class labels from YOLO results
        cv2.putText(frame, boxes['names'].iloc[0], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Object Tracking', frame)

    # Update for next frame
    prev_frame = frame.copy()
    prev_boxes = tracked_boxes

    # Handle user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from pykalman import KalmanFilter
import numpy as np

# Function to calculate optical flow using Farneback method (consider alternatives for efficiency)
def calculate_optical_flow(prev_frame, curr_frame):
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Experiment with different parameters for FarnebackOpticalFlow (5, 13, 15)
    # You might consider alternative optical flow algorithms for performance
    farneback = cv2.FarnebackOpticalFlow_create(5, 13, 15, 0.5, 5.0, 1.2, 0, 0, 1.0, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = farneback.calcOpticalFlow(gray_prev, gray_curr, None)

    return flow

# Function to update Kalman filter with detection and optical flow
def update_kalman_filter(kf, detection, flow):
    # Extract relevant information from detection (consider your specific detection format)
    x, y, w, h = detection

    # Create measurement vector based on detection and optical flow (adjust based on your needs)
    z = np.array([x + flow[y, x][0], y + flow[y, x][1]])  # Account for movement

    # Update Kalman filter
    kf = kf.update(z)

    return kf

# Load YOLO model (replace with your preferred model and path)
model = cv2.dnn.readNetFromDarkNet("yolov8n.cfg", "yolov8n.weights")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # Enable CUDA if available

# Load video capture
cap = cv2.VideoCapture("qwe.mp4")

# Define Kalman filter parameters (adjust based on object dynamics and noise)
dt = 1.0  # Time step (assuming constant frame rate)
kf = KalmanFilter(dim_x=4, dim_z=2, dt=dt)
transition_matrix = np.array([[1, dt, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, dt],
                             [0, 0, 0, 1]])
observation_matrix = np.array([[1, 0],
                               [0, 1]])
process_noise_cov = np.eye(4) * 1e-3  # Adjust covariance based on process noise
measurement_noise_cov = np.eye(2) * 1e-1  # Adjust covariance based on measurement noise
kf.transition_matrices = transition_matrix
kf.observation_matrices = observation_matrix
kf.process_noise_cov = process_noise_cov
kf.measurement_noise_cov = measurement_noise_cov

# Track information (replace with appropriate data structures for multiple objects)
track_id = 0
detections = []
boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for YOLO
    # (Your preprocessing steps here, e.g., resizing, normalization)

    # Run YOLO object detection
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward()

    # Parse YOLO detections (extract relevant information based on your model's output)
    new_detections = []
    for i in range(outputs.shape[0]):
        confidence = outputs[i, 5:]
        class_id = int(outputs[i, 4])
        if confidence[class_id] > 0.5:
            center_x = int(outputs[i, 0] * frame.shape[1])
            center_y = int(outputs[i, 1] * frame.shape[0])
                        # Extract bounding box coordinates (adjust based on your model's output format)
            x = center_x - int(outputs[i, 2] * frame.shape[1] / 2)
            y = center_y - int(outputs[i, 3] * frame.shape[0] / 2)
            w = int(outputs[i, 2] * frame.shape[1])
            h = int(outputs[i, 3] * frame.shape[0])
            new_detections.append((x, y, w, h))

    # Track association with Kalman filter
    if len(detections) > 0 and len(new_detections) > 0:
        # Efficient data association (consider Hungarian algorithm for multiple objects)
        iou_matrix = np.zeros((len(detections), len(new_detections)))
        for i in range(len(detections)):
            for j in range(len(new_detections)):
                # Calculate Intersection-over-Union (IoU) between detections and new detections
                box1 = detections[i]
                box2 = new_detections[j]
                xmin1, ymin1, xmax1, ymax1 = box1
                xmin2, ymin2, xmax2, ymax2 = box2
                intersection_area = max(0, (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) -
                                       (max(xmin1, xmin2) - min(xmax1, xmax2)) * (max(ymin1, ymin2) - min(ymax1, ymax2)))
                union_area = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection_area
                iou_matrix[i, j] = float(intersection_area) / float(union_area)

        # Assign detections to tracks using maximum IoU (modify for more complex association)
        assignments = np.zeros((len(detections),), dtype=np.int)
        for i in range(len(detections)):
            assignments[i] = np.argmax(iou_matrix[i, :])

        # Update Kalman filters and boxes
        updated_boxes = []
        for i in range(len(detections)):
            track_id = assignments[i]
            if track_id >= 0:
                # Update Kalman filter with detection and optical flow (if available)
                kf = update_kalman_filter(kf, new_detections[track_id], flow)
                # Predicted state from Kalman filter
                predicted_x, predicted_y = kf.state[0], kf.state[1]
                updated_boxes.append((int(predicted_x), int(predicted_y), w, h))
            else:
                # Lost track, handle appropriately (e.g., remove track)
                pass
    else:
        # No detections or new detections, handle appropriately (e.g., predict using Kalman filter)
        updated_boxes = []

    # Update detections and boxes for next frame
    detections = updated_boxes
    boxes = detections

    # Visualize tracking results (modify as needed)
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Object Tracking', frame)

    # Calculate optical flow for the next frame (consider efficiency for long videos)
    prev_frame = frame.copy()

    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
