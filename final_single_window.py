import cv2
import time
import numpy as np
from Camera_video_mini import density
from speed_calc_mini import speed_estimation1
from collections import defaultdict
from ultralytics import YOLO

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

# Create window to display analyzed frames
cv2.namedWindow('Analyzed Frame', cv2.WINDOW_NORMAL)

prev=0
frame_no=0
skip_var=10

# Store the track history
track_history = defaultdict(lambda: [])

# Store the direction vectors for each track
direction_vectors = defaultdict(list)

# Number of frames to consider for calculating the most common direction
num_frames_to_consider = 1

time10=0
time1=0
desired_fps = cap.get(cv2.CAP_PROP_FPS)
skip_var=1
# Process each frame from the camera or video
while True:
    frame_no+=1
    success, frame = cap.read()
    big_frame=frame
    if not success:
        print("End of video or failed to capture frame")
        break
    if success and (frame_no%skip_var==0 or frame_no==1):
        # Resize frame
        frame = cv2.resize(frame, (resize_width, resize_height))
        
        overlay,num_people=density(frame)
        # Overlay density map on frame
        alpha = 0.5
        
        # Combine frame and overlay
        combined = cv2.addWeighted(frame, 1, overlay, alpha, 0)
        optical_res='frame analysing'
        combined_image1 = cv2.vconcat([frame, combined])
        if frame_no==1:
            prev=frame
            heatmap_frame=frame
            anotated_frame=frame
            results = model.track(frame, persist=True)
            #_,track_history, direction_vectors=track(frame,track_history,direction_vectors,model)
            
        else:
            heatmap_frame,optical_res,speeds,=speed_estimation1(speeds, prev, frame, frame_no)
            prev=frame
            results = model.track(frame, persist=True)
            # if (frame_no%skip_var==0 or frame_no==1):
            #     anotated_frame,track_history, direction_vectors=track(big_frame,track_history,direction_vectors,model)
            #     anotated_frame = cv2.resize(anotated_frame, (resize_width, resize_height))
        
        width, height = resize_width,resize_height  # dimensions of the frame
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)  # creating a black frame

        # Step 2: Add text on the frame
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
            # Check if there are detections in the current frame
            # Get the boxes and track IDs
            # Check if there are detections in the current frame
           if results is not None and len(results) > 0:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            class_labels = results[0].pred[0].get("class").int().cpu().tolist()  # Assuming class labels are available
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks and calculate direction vectors for person detections only
            for box, class_label, track_id in zip(boxes, class_labels, track_ids):
                if class_label == 0:  # Filter only person detections
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

            # Calculate the average direction vectors across all person tracks
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
            print("emptyq")
        
        combined_image2 = cv2.vconcat([heatmap_frame, annotated_frame])
        
        combined_image = cv2.hconcat([combined_image1, combined_image2])
        
    
    
    # Display the analyzed frame
    cv2.imshow('Analyzed Frame', combined_image)
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


def twoon(input_source):
        # Set resize parameters
    resize_width = 640  # Change this according to your preference
    resize_height = 480  # Change this according to your preference


    speeds=[]

    # Choose input source
     

    # Open camera capture or video file
    if isinstance(input_source, int):  # Camera input
        cap = cv2.VideoCapture(input_source)
    else:
        cap = cv2.VideoCapture(input_source)
        
    # # Set capture properties
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create window to display analyzed frames
    cv2.namedWindow('Analyzed Frame', cv2.WINDOW_NORMAL)

    prev=0
    frame_no=0
    skip_var=10

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Store the direction vectors for each track
    direction_vectors = defaultdict(list)

    # Number of frames to consider for calculating the most common direction
    num_frames_to_consider = 1

    time10=0
    time1=0
    desired_fps = cap.get(cv2.CAP_PROP_FPS)
    skip_var=1
    # Process each frame from the camera or video
    while True:
        frame_no+=1
        success, frame = cap.read()
        big_frame=frame
        if not success:
            print("End of video or failed to capture frame")
            break
        if success and (frame_no%skip_var==0 or frame_no==1):
            # Resize frame
            frame = cv2.resize(frame, (resize_width, resize_height))
            
            overlay,num_people=density(frame)
            # Overlay density map on frame
            alpha = 0.5
            
            # Combine frame and overlay
            combined = cv2.addWeighted(frame, 1, overlay, alpha, 0)
            optical_res='frame analysing'
            combined_image1 = cv2.vconcat([frame, combined])
            if frame_no==1:
                prev=frame
                heatmap_frame=frame
                anotated_frame=frame
                #_,track_history, direction_vectors=track(frame,track_history,direction_vectors,model)
                
            else:
                heatmap_frame,optical_res=speed_estimation1(speeds, prev, frame, frame_no)
                prev=frame
                # if (frame_no%skip_var==0 or frame_no==1):
                #     anotated_frame,track_history, direction_vectors=track(big_frame,track_history,direction_vectors,model)
                #     anotated_frame = cv2.resize(anotated_frame, (resize_width, resize_height))
            
            width, height = resize_width,resize_height  # dimensions of the frame
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)  # creating a black frame

            # Step 2: Add text on the frame
            text = f'People: {num_people}'+'\n'+ optical_res  # text to be added
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5  # initial font scale
            font_thickness = 1
            text_color = (255, 255, 255)  # white color

            # Calculate the text size
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            while text_size[0] > resize_width - 20 or text_size[1] > resize_height - 20:  # Adjust font size to fit the text
                font_scale -= 0.05
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Calculate text position
            text_x = (resize_width - text_size[0]) // 2  # calculate x-coordinate for centering text
            text_y = (resize_height + text_size[1]) // 2  # calculate y-coordinate for centering text
            cv2.putText(black_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
            
            
            combined_image2 = cv2.vconcat([heatmap_frame, black_frame])
            
            combined_image = cv2.hconcat([combined_image1, combined_image2])
            
        
        
        # Display the analyzed frame
        cv2.imshow('Analyzed Frame', combined_image)
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
