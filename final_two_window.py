import cv2
import os
import time
import numpy as np
from Camera_video_mini import density
from speed_calc_mini import speed_estimation1
from collections import defaultdict
from ultralytics import YOLO
import csv
from datetime import datetime

model=YOLO('yolov8n.pt')



# Function to choose camera or video input
def choose_input():
    while True:
        # choice='video'
        choice = input("Choose input source (camera or video): ").lower()
        if choice == 'camera':
            return 0  # Return camera index
        elif choice == 'video':
            # video_path = r'2.mov' 
            video_path = input("Enter video file path: ")
            return video_path  # Return video file pathv
        else:
            print("Invalid input. Please choose 'camera' or 'video'.")



# Set resize parameters
resize_width = 640  # Change this according to your preference
resize_height = 480  # Change this according to your preference


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

# Store the track history
track_history = defaultdict(lambda: [])
direction_vectors=defaultdict(lambda: [])

# # Create window to display analyzed frames
# cv2.namedWindow('Analyzed Frame', cv2.WINDOW_NORMAL)

prev=0
frame_no=0
skip_var=10

# Store the track history
track_history = defaultdict(lambda: [])

# Store the direction vectors for each track
direction_vectors = defaultdict(list)

# Number of frames to consider for calculating the most common directioncamer
num_frames_to_consider = 1

time10=0
time1=0
desired_fps = cap.get(cv2.CAP_PROP_FPS)
skip_var=1

# Define the column names
field_names = ["Time","NumOfPeople", "Optical Flow"]

# Name of the CSV file
csv_file = os.path.join("Recorded_Data",datetime.now().strftime('%Y%m_%H%M%S')+'.csv')

# Writing column names to CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=field_names)

    # Write the header
    writer.writeheader()

# Process each frame from the camera or video
while True:
    frame_no+=1
    success, frame = cap.read()
    annotated_frame=big_frame=frame
    num_people=0
    avg=0
    if not success:
        print("End of video or failed to capture frame")
        break
    if success and (frame_no%skip_var==0 or frame_no==1):
        # Resize frame
        frame = cv2.resize(frame, (resize_width, resize_height))
        
        overlay,num_people=density(frame)
        # Overlay density map on frame
        alpha = 0.5
        
        # Add text to images
        text1 = "Orginal"
        text2 = "Density Heatmap"
        text3 = "Motion Heatmap"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(frame.shape[0], frame.shape[1]) / 1000  # Adjust this scaling factor as needed
        font_thickness = 2
        text_color = (255, 255, 255)
        
        # Calculate text size
        text_size1 = cv2.getTextSize(text1, font, font_scale, font_thickness)[0]
        text_size2 = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]
        text_size3 = cv2.getTextSize(text3, font, font_scale, font_thickness)[0]

        # Position text at the bottom-left corner of the image
        text_position1 = (15, frame.shape[0] - int(15 * font_scale))
        text_position2 = (15, frame.shape[0] - int(15 * font_scale))
        text_position3 = (15, frame.shape[0] - int(15 * font_scale))
        
        # Combine frame and overlay
        density_map = cv2.addWeighted(frame, 1, overlay, alpha, 0)
        optical_res='frame analysing'
        
         # Put text on images
        cv2.putText(frame, text1, text_position1, font, font_scale, text_color, font_thickness)
        cv2.putText(density_map, text2, text_position2, font, font_scale, text_color, font_thickness)
        
        combined_image1 = cv2.vconcat([frame, density_map])
        if frame_no==1:
            prev=frame
            heatmap_frame=frame
            anotated_frame=frame
            results = model.track(big_frame, persist=True, classes=0)
            #_,track_history, direction_vectors=track(frame,track_history,direction_vectors,model)
            
        else:
            heatmap_frame,optical_res,speeds,avg=speed_estimation1(speeds, prev, frame, frame_no)
            prev=frame
            try:
                results = model.track(big_frame, persist=True, classes=0)
            except:
                print('nah')
            # if (frame_no%skip_var==0 or frame_no==1):
            #     anotated_frame,track_history, direction_vectors=track(big_frame,track_history,direction_vectors,model)
            #     anotated_frame = cv2.resize(anotated_frame, (resize_width, resize_height))
        
        cv2.putText(heatmap_frame, text3, text_position3, font, font_scale, text_color, font_thickness)
        width, height = resize_width,resize_height  # dimensions of the frame
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)  # creating a black frame

        # Add text on the frame
        text = f'People: {num_people}'+' '+ optical_res  # text to be added
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # initial font scale
        font_thickness = 1
        text_color = (255, 255, 255)  # white color
        print(text)
        # Calculate the text size
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        while text_size[0] > resize_width - 20 or text_size[1] > resize_height - 20:  # Adjust font size to fit the text
            font_scale -= 0.05
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Calculate text position
        text_x = (resize_width - text_size[0]) // 2  # calculate x-coordinate for centering text
        text_y = (resize_height + text_size[1]) // 2  # calculate y-coordinate for centering text
        cv2.putText(black_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        try:
            if results is not None and len(results) > 0:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # print(results)
                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks and calculate direction vectors
                for box, track_id,r in zip(boxes, track_ids,results):  
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
                        scaled_vector = average_direction * 500 # Adjust the scaling factor as needed

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
            annotated_frame=big_frame
            print("emptyq")
        
        combined_image2 = cv2.vconcat([heatmap_frame, black_frame])
        
        combined_image = cv2.hconcat([combined_image1, combined_image2])
    

        
    # Create windows with resizable flag
    cv2.namedWindow('Density and optical flow', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Direction Detection', cv2.WINDOW_NORMAL)
    
    # Display the analyzed frame
    cv2.imshow('Density and optical flow', combined_image)
    cv2.imshow('Direction Detection', annotated_frame)
    
    data_to_add = [
        {"Time":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
         "NumOfPeople": num_people, "Optical Flow": avg}
    ]

    # Appending data to CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)

        # Write each row
        for row in data_to_add:
            writer.writerow(row)

    print("Data added to the CSV file.")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
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
        if skip_var<=0: 
            skip_var=1
        print(frames_per_second,' ',avg_time_per_10_frames)
        print("Frames to skip for real-time processing:", skip_var)
        print("-----------------------------------")
    print(frame_no)
    
# Release resources
cap.release()
cv2.destroyAllWindows()