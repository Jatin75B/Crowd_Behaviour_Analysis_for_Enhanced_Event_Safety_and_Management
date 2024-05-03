import cv2
import time
import numpy as np
from Camera_video_mini import density
from speed_calc_mini import speed_estimation1
from collections import defaultdict, Counter
from movement.move_track import track
from ultralytics import YOLO

# model=YOLO('yolov8n.pt')

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
